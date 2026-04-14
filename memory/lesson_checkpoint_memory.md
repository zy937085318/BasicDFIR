# checkpoint 显存优化说明

## 功能
梯度检查点（gradient checkpointing）是一种**节省显存**的技术。

## 工作原理
- 前向传播时不保存中间激活值
- 反向传播时重新计算这些激活值
- 用计算时间换显存空间

## 配置
```yaml
use_checkpoint: true  # 启用以节省显存
```

## 注意事项
- 会增加训练时间（需要重计算）
- 适合大模型或显存受限场景
- DinoATD 模型中默认开启此选项

## 相关文件
- `/Users/ybb/code zoo/BasicDFIR/basicsr/archs/dinoatd_arch.py`
