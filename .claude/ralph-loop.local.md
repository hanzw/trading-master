---
active: true
iteration: 1
max_iterations: 0
completion_promise: "ALL 30 MODELS IMPLEMENTED AND 1000 TESTS PASSING"
started_at: "2026-03-18T00:00:00Z"
---

# 永续财富优化引擎 — 每次迭代执行以下步骤

## Step 1: 状态评估
cd D:/trading-master && git log --oneline -3 && python -m pytest tests/ -q

## Step 2: 评审团 (3个专家sub-agent)
启动3个评审agent，每人1段评审 + 1个建议:
- 华尔街交易大师: 最能保护$1M组合的下一个功能
- 量化教授: 从模型列表选下一个最重要的模型实现
- AI架构师: 最能提升可靠性的改进

## Step 3: 开发团队 (5-10个agent并行)
根据评审建议 + 模型覆盖度列表，启动开发agent:
- 2个agent实现下一个量化模型 (Black-Litterman, Fama-French, Monte Carlo, DCF, GARCH, HRP, Risk Parity等)
- 2个agent实现投资分析功能 (行业轮动, 资金流向, 期权情绪, 多时间框架技术面等)
- 1个agent改进工程质量 (测试, 性能, 可靠性)

## Step 4: 验证并提交
python -m pytest tests/ -v && git add -A && git commit && git push

## Step 5: 三层防停检查
1. 评审还有建议？→ 继续
2. 模型列表还有未实现的？→ 继续  
3. 组合有新优化机会？→ 继续
永远继续下一次迭代。
