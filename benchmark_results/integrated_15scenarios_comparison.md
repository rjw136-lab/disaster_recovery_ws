# Disaster Recovery 15-Scenario Integrated Benchmark Comparison
- Generated at: `2026-11-24 17:56:01`
- Source root: `/home/jwroh/ws/disaster_recovery_ws/benchmark_results`

## 1. Scenario-wise Success Rate Comparison (15 Scenarios × 5 Strategies)
| Scenario | Baseline | Phase 2-B | Phase 3 | Phase 4 (Sim) | Phase 4 (Real) | Best Strategy |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| collapsed |  10.0% (1/10) |  70.0% (7/10) |  90.0% (9/10) | 100.0% (10/10) | 100.0% (10/10) | Phase 4 (Sim), Phase 4 (Real) |
| collapsed_heavy |   0.0% (0/10) |  90.0% (9/10) |  80.0% (8/10) |  90.0% (9/10) |  90.0% (9/10) | Phase 2-B, Phase 4 (Sim), Phase 4 (Real) |
| tunnel |  10.0% (1/10) | 100.0% (10/10) |  90.0% (9/10) | 100.0% (10/10) | 100.0% (10/10) | Phase 2-B, Phase 4 (Sim), Phase 4 (Real) |
| house |   0.0% (0/10) |  40.0% (4/10) |  40.0% (4/10) |  60.0% (6/10) |  50.0% (5/10) | Phase 4 (Sim) |
| maze |   0.0% (0/10) |  90.0% (9/10) | 100.0% (10/10) | 100.0% (10/10) | 100.0% (10/10) | Phase 3, Phase 4 (Sim), Phase 4 (Real) |
| collapsed_hard |   0.0% (0/10) |  70.0% (7/10) |  80.0% (8/10) | 100.0% (10/10) |  90.0% (9/10) | Phase 4 (Sim) |
| collapsed_heavy_hard |   0.0% (0/10) |  30.0% (3/10) |  30.0% (3/10) |  50.0% (5/10) |  50.0% (5/10) | Phase 4 (Sim), Phase 4 (Real) |
| tunnel_hard |   0.0% (0/10) | 100.0% (10/10) | 100.0% (10/10) | 100.0% (10/10) | 100.0% (10/10) | Phase 2-B, Phase 3, Phase 4 (Sim), Phase 4 (Real) |
| house_hard |   0.0% (0/10) |  20.0% (2/10) |  20.0% (2/10) |  50.0% (5/10) |  60.0% (6/10) | Phase 4 (Real) |
| maze_hard |   0.0% (0/10) | 100.0% (10/10) | 100.0% (10/10) | 100.0% (10/10) | 100.0% (10/10) | Phase 2-B, Phase 3, Phase 4 (Sim), Phase 4 (Real) |
| u_pocket_trap |   0.0% (0/10) | 100.0% (10/10) | 100.0% (10/10) | 100.0% (10/10) | 100.0% (10/10) | Phase 2-B, Phase 3, Phase 4 (Sim), Phase 4 (Real) |
| slip_chicane |   0.0% (0/10) |  60.0% (6/10) |  50.0% (5/10) |  60.0% (6/10) |  60.0% (6/10) | Phase 2-B, Phase 4 (Sim), Phase 4 (Real) |
| dynamic_gauntlet |   0.0% (0/10) |  40.0% (4/10) |  70.0% (7/10) |  60.0% (6/10) |  30.0% (3/10) | Phase 3 |
| dynamic_crossfire_alley |   0.0% (0/10) |   0.0% (0/10) |   0.0% (0/10) |   0.0% (0/10) |  40.0% (4/10) | Phase 4 (Real) |
| smoke_lidar_corruption |   0.0% (0/10) |  80.0% (8/10) |  70.0% (7/10) | 100.0% (10/10) | 100.0% (10/10) | Phase 4 (Sim), Phase 4 (Real) |
| **OVERALL (Micro)** | **  1.3% (2/150)** | ** 66.0% (99/150)** | ** 68.0% (102/150)** | ** 78.0% (117/150)** | ** 78.0% (117/150)** | - |
| **MACRO AVG (Scen)** | **  1.3%** | ** 66.0%** | ** 68.0%** | ** 78.0%** | ** 78.0%** | - |

## 2. Overall Multi-Metric Summary (150 Trials per Strategy)
| Strategy | Trials | Success (%) | Arrival (%) | Collision-Free (%) | Succ Time (s) | Min Debris (m) | Term Err (cm) | vs Base (W/T/L) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **Baseline** | 150/150 |   1.3% (2/150) |  53.3% |   8.0% | 92.80 | 0.276 | 2.92 | - |
| **Phase 2-B** | 150/150 |  66.0% (99/150) |  97.3% |  66.0% | 78.42 | 0.529 | 9.04 | 14W / 1T / 0L |
| **Phase 3** | 150/150 |  68.0% (102/150) |  95.3% |  68.0% | 79.98 | 0.523 | 9.20 | 14W / 1T / 0L |
| **Phase 4 (Sim)** | 150/150 |  78.0% (117/150) |  92.0% |  79.3% | 97.98 | 0.546 | 9.12 | 14W / 1T / 0L |
| **Phase 4 (Real)** | 150/150 |  78.0% (117/150) |  92.7% |  78.7% | 90.54 | 0.579 | 9.32 | 15W / 0T / 0L |
