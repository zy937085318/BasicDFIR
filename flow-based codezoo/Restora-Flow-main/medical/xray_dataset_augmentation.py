import os
import SimpleITK as sitk
import numpy as np

from medical_data_augment_tool.datasources.cached_image_datasource import CachedImageDataSource
from medical_data_augment_tool.datasources.image_datasource import ImageDataSource
from medical_data_augment_tool.datasources.landmark_datasource import LandmarkDataSource
from medical_data_augment_tool.datasets.reference_image_transformation_dataset import ReferenceTransformationDataset
from medical_data_augment_tool.generators.image_generator import ImageGenerator
from medical_data_augment_tool.generators.landmark_generator import LandmarkGenerator, LandmarkGeneratorHeatmap
from medical_data_augment_tool.iterators.id_list_iterator import IdListIterator
from medical_data_augment_tool.transformations.intensity.np.normalize import normalize_robust
from medical_data_augment_tool.transformations.intensity.np.shift_scale_clamp import ShiftScaleClamp
from medical_data_augment_tool.transformations.spatial import composite, rotation, scale, translation
from medical_data_augment_tool.utils.sitk_image import reduce_dimension


class XRayDatasetAugmentation:
    def __init__(self, dataset_folder_path, image_size, image_extension, train, image_ids_file,
                 heatmap_sigma, num_landmarks, landmarks_file_name='landmark_annotations.csv',
                 random_augmentation=True, save_debug_images=False) -> None:

        self.dataset_folder = dataset_folder_path
        self.images_folder = os.path.join(self.dataset_folder, 'images')

        self.dim = 2
        self.image_size = image_size
        self.image_extension = image_extension
        self.data_format = 'channels_first'

        self.landmarks_file_path = os.path.join(self.dataset_folder, 'setup', landmarks_file_name)
        self.num_landmarks = num_landmarks
        self.heatmap_size = image_size
        self.heatmap_scale = 100
        self.downsampling_factor = self.image_size[0] / self.heatmap_size[0]
        self.heatmap_sigma = heatmap_sigma
        self.save_debug_images = save_debug_images
        self.image_ids_file = image_ids_file
        self.train = train
        self.random = random_augmentation

        if self.train:
            self.dataset = self.dataset_train()
        else:
            self.dataset = self.dataset_val()

    def spatial_transformation(self):
        """
        The spatial image transformation without random augmentation.
        :return: The transformation.
        """
        return composite.Composite(self.dim,
                                   [translation.InputCenterToOrigin(self.dim),
                                    scale.FitFixedAr(self.dim, self.image_size),
                                    translation.OriginToOutputCenter(self.dim, self.image_size)]
                                   )

    def spatial_transformation_augmented(self):
        """
        The spatial image transformation with random augmentation.
        :return: The transformation.
        """
        return composite.Composite(self.dim,
                                   [translation.InputCenterToOrigin(self.dim),
                                    scale.FitFixedAr(self.dim, self.image_size),
                                    translation.Random(self.dim, [5, 5]),  # [10, 10]
                                    rotation.Random(self.dim, [0.2, 0.2]),
                                    translation.OriginToOutputCenter(self.dim, self.image_size),
                                    # deformation.Output(self.dim, [5, 5], 20, self.image_size)
                                    ]
                                   )

    @staticmethod
    def intensity_postprocessing(image):
        """
        Intensity postprocessing.
        :param image: The np input image.
        :return: The processed image.
        """
        normalized = normalize_robust(image)
        return normalized

    @staticmethod
    def intensity_postprocessing_augmented(image):
        """
        Intensity postprocessing. Random augmentation version.
        :param image: The np input image.
        :return: The processed image.
        """
        normalized = normalize_robust(image)
        # normalized = normalize_robust(image, consideration_factors=(0.01, 0.01))
        return ShiftScaleClamp(random_shift=0.15, random_scale=0.15)(normalized)

    def data_sources(self, cached):
        """
        Returns the data sources that load data.
        :param cached: If true, use a CachedImageDataSource instead of an ImageDataSource.
        :return: A dict of data sources('image_datasource:' ImageDataSource that loads the image files,
                                        'landmarks_datasource:' LandmarkDataSource that loads the landmark coordinates.)
        """
        if cached:
            image_datasource = CachedImageDataSource(self.images_folder,
                                                     '',
                                                     '',
                                                     self.image_extension,
                                                     preprocessing=reduce_dimension,
                                                     set_identity_spacing=True,
                                                     cache_maxsize=8192,
                                                     sitk_pixel_type=sitk.sitkFloat32)   # 16384
        else:
            image_datasource = ImageDataSource(self.images_folder,
                                               '',
                                               '',
                                               self.image_extension,
                                               preprocessing=reduce_dimension,
                                               set_identity_spacing=True,
                                               sitk_pixel_type=sitk.sitkFloat32)

        landmarks_datasource = LandmarkDataSource(self.landmarks_file_path,
                                                  self.num_landmarks,
                                                  self.dim)

        return {'reference_image': image_datasource,
                'landmarks_datasource': landmarks_datasource
                }

    def data_generators(self, image_post_processing_np=None):
        """
        Returns the data generators that process one input. See data_sources() for dict values.
        :param image_post_processing_np: The np postprocessing function for the image data generator.
        :return: A dict of data generators.
        """
        image_generator = ImageGenerator(self.dim,
                                         self.image_size,
                                         post_processing_np=image_post_processing_np,
                                         interpolator='linear',
                                         resample_default_pixel_value=0,
                                         data_format=self.data_format,
                                         resample_sitk_pixel_type=sitk.sitkFloat32,
                                         np_pixel_type=np.float32)

        if self.downsampling_factor == 1:
            heatmap_post_transformation = None
        else:
            heatmap_post_transformation = scale.Fixed(self.dim, self.downsampling_factor)

        landmark_heatmap_generator = LandmarkGeneratorHeatmap(self.dim,
                                                              self.heatmap_size,
                                                              [1] * self.dim,
                                                              self.heatmap_sigma,
                                                              scale_factor=self.heatmap_scale,
                                                              normalize_center=True,
                                                              data_format=self.data_format,
                                                              post_transformation=heatmap_post_transformation)

        landmark_2d_points_generator = LandmarkGenerator(self.dim,
                                                         self.heatmap_size,
                                                         [1] * self.dim)

        return {'transformed_image': image_generator,
                'landmarks_heatmap': landmark_heatmap_generator,
                'landmarks_2d_points': landmark_2d_points_generator
                }

    @staticmethod
    def data_generator_sources():
        """
        Returns a dict that defines the connection between datasources and datagenerator parameters for their get() function.
        :return: A dict.
        """
        return {'transformed_image': {'image': 'reference_image'},
                'landmarks_heatmap': {'landmarks': 'landmarks_datasource'},
                'landmarks_2d_points': {'landmarks': 'landmarks_datasource'}}

    def dataset_train(self):
        """
        Returns the training dataset. Random augmentation is performed.
        :return: The training dataset.
        """
        data_sources = self.data_sources(True)
        data_generator_sources = self.data_generator_sources()
        data_generators = self.data_generators(self.intensity_postprocessing_augmented)
        if self.random:
            image_transformation = self.spatial_transformation_augmented()
        else:
            image_transformation = self.spatial_transformation()
        iterator = IdListIterator(self.image_ids_file,
                                  random=self.random,
                                  keys=['image_id'])

        dataset = ReferenceTransformationDataset(dim=self.dim,
                                                 reference_datasource_keys={'image': 'reference_image'},
                                                 reference_transformation=image_transformation,
                                                 data_sources=data_sources,
                                                 data_generators=data_generators,
                                                 data_generator_sources=data_generator_sources,
                                                 iterator=iterator,
                                                 debug_image_folder='debug_train_images' if self.save_debug_images
                                                 else None)

        return dataset

    def dataset_val(self):
        """
        Returns the validation dataset. No random augmentation is performed.
        :return: The validation dataset.
        """
        data_sources = self.data_sources(False)
        data_generator_sources = self.data_generator_sources()
        data_generators = self.data_generators(self.intensity_postprocessing)
        image_transformation = self.spatial_transformation()
        iterator = IdListIterator(self.image_ids_file,
                                  random=False,
                                  keys=['image_id'])
        dataset = ReferenceTransformationDataset(dim=self.dim,
                                                 reference_datasource_keys={'image': 'reference_image'},
                                                 reference_transformation=image_transformation,
                                                 data_sources=data_sources,
                                                 data_generators=data_generators,
                                                 data_generator_sources=data_generator_sources,
                                                 iterator=iterator,
                                                 debug_image_folder='debug_val_images' if self.save_debug_images
                                                 else None)
        return dataset

    def get_data(self, image_id=None):
        if image_id is None:
            dic = {'image_id': image_id}
            dataset_entry = self.dataset.get(dic)
        else:
            dataset_entry = self.dataset.get_next()

        data_sources = dataset_entry['datasources']
        data_generators = dataset_entry['generators']
        transformations = dataset_entry['transformations']

        reference_image = data_sources['reference_image']
        image_transform = transformations['transformed_image']
        transformed_image = data_generators['transformed_image']
        corresponding_heatmap = data_generators['landmarks_heatmap']
        transformed_landmarks = data_generators['landmarks_2d_points']

        # swap x and y coordinates (xray)
        transformed_landmarks[:, [1, 2]] = transformed_landmarks[:, [2, 1]]

        return reference_image, image_transform, transformed_image, corresponding_heatmap, transformed_landmarks
