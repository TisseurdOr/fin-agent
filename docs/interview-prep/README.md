# fin-agent 面试准备材料

本目录配合根目录面试问题清单使用，按复习顺序阅读：

| 文件 | 内容 | 建议时间 |
|------|------|----------|
| [01-elevator-pitch.md](01-elevator-pitch.md) | 1 分钟 / 5 分钟项目介绍 + 架构口述 + Demo 路径 | 30 min |
| [02-rag-story.md](02-rag-story.md) | RAG 迭代数据、消融结论、高频追问答法 | 45 min |
| [03-weaknesses.md](03-weaknesses.md) | 15 个已知弱点 + 改进方案 | 30 min |
| [04-deep-dive.md](04-deep-dive.md) | `/api/askRAG` 与 `/api/analyze` 逐步拆解 | 45 min |

## 复习 Checklist

- [ ] 能脱稿讲 1 分钟版
- [ ] 能脱稿讲 5 分钟版（含画架构图）
- [ ] 背出 RAG 关键数字：88.9% → 100%，B1 生产 2.3s
- [ ] 能逐步讲清 askRAG 和 analyze 两条链路
- [ ] 每个弱点都能答「现状 + 改进」
- [ ] 跑过 `python test_recall.py` 和 `pytest tests/`

## 相关文件

- 完整问题列表：见 Cursor plan「面试问题清单」
- RAG 技术文档：[../rag-improvement-guide.md](../rag-improvement-guide.md)
- 召回测试脚本：[../../test_recall.py](../../test_recall.py)
