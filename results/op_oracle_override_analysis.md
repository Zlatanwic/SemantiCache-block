# OP-SieveKV Oracle Dataset Analysis

- Dataset: `results/op_oracle_dataset_pilot.pt`
- Rows: 7896
- Oracle rows: 1076
- Entropy threshold: 0.2880
- Divergence threshold: 0.6137

## KV-TIP Quadrants

- not_oracle: 6820
- Q1_confident_aligned: 364
- Q2_uncertain_aligned: 440
- Q4_uncertain_wrong: 100
- Q3_confident_wrong: 172

## Decision Types

- not_oracle: 6820
- aligned_keep: 352
- over_keep: 724

## Oracle By Budget

### Budget 0.05

- Rows: 269
- Hard labels: 88
- Label mass: 134.3877
- Mean oracle delta: 0.3235
- Mean oracle label: 0.4992
- Reasons: `{'oracle_keep': 88, 'oracle_drop': 181}`

### Budget 0.1

- Rows: 269
- Hard labels: 88
- Label mass: 134.3877
- Mean oracle delta: 0.3235
- Mean oracle label: 0.4992
- Reasons: `{'oracle_keep': 88, 'oracle_drop': 181}`

### Budget 0.2

- Rows: 269
- Hard labels: 88
- Label mass: 134.3877
- Mean oracle delta: 0.3235
- Mean oracle label: 0.4992
- Reasons: `{'oracle_keep': 88, 'oracle_drop': 181}`

### Budget 0.3

- Rows: 269
- Hard labels: 88
- Label mass: 134.3877
- Mean oracle delta: 0.3235
- Mean oracle label: 0.4992
- Reasons: `{'oracle_keep': 88, 'oracle_drop': 181}`

## Top Q3 Confident-Wrong Rows

- `{'row_index': 3733, 'task': 'multi_needle', 'budget': 0.3, 'num_needles': 2, 'segment_start': 1746, 'segment_end': 1754, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9898021817207336, 'oracle_label': 0.014299706555902958, 'oracle_delta': -0.2886490821838379, 'retention_entropy': 0.08210171094932298, 'oracle_policy_divergence': 0.9755024751648307}`
- `{'row_index': 3911, 'task': 'multi_needle', 'budget': 0.2, 'num_needles': 2, 'segment_start': 1746, 'segment_end': 1754, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9898021817207336, 'oracle_label': 0.014299706555902958, 'oracle_delta': -0.2886490821838379, 'retention_entropy': 0.08210171094932298, 'oracle_policy_divergence': 0.9755024751648307}`
- `{'row_index': 4089, 'task': 'multi_needle', 'budget': 0.1, 'num_needles': 2, 'segment_start': 1746, 'segment_end': 1754, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9898021817207336, 'oracle_label': 0.014299706555902958, 'oracle_delta': -0.2886490821838379, 'retention_entropy': 0.08210171094932298, 'oracle_policy_divergence': 0.9755024751648307}`
- `{'row_index': 4267, 'task': 'multi_needle', 'budget': 0.05, 'num_needles': 2, 'segment_start': 1746, 'segment_end': 1754, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9898021817207336, 'oracle_label': 0.014299706555902958, 'oracle_delta': -0.2886490821838379, 'retention_entropy': 0.08210171094932298, 'oracle_policy_divergence': 0.9755024751648307}`
- `{'row_index': 3731, 'task': 'multi_needle', 'budget': 0.3, 'num_needles': 2, 'segment_start': 1714, 'segment_end': 1729, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.993743360042572, 'oracle_label': 0.030423128977417946, 'oracle_delta': -0.22693252563476562, 'retention_entropy': 0.05479921018626801, 'oracle_policy_divergence': 0.9633202310651541}`
- `{'row_index': 3909, 'task': 'multi_needle', 'budget': 0.2, 'num_needles': 2, 'segment_start': 1714, 'segment_end': 1729, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.993743360042572, 'oracle_label': 0.030423128977417946, 'oracle_delta': -0.22693252563476562, 'retention_entropy': 0.05479921018626801, 'oracle_policy_divergence': 0.9633202310651541}`
- `{'row_index': 4087, 'task': 'multi_needle', 'budget': 0.1, 'num_needles': 2, 'segment_start': 1714, 'segment_end': 1729, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.993743360042572, 'oracle_label': 0.030423128977417946, 'oracle_delta': -0.22693252563476562, 'retention_entropy': 0.05479921018626801, 'oracle_policy_divergence': 0.9633202310651541}`
- `{'row_index': 4265, 'task': 'multi_needle', 'budget': 0.05, 'num_needles': 2, 'segment_start': 1714, 'segment_end': 1729, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.993743360042572, 'oracle_label': 0.030423128977417946, 'oracle_delta': -0.22693252563476562, 'retention_entropy': 0.05479921018626801, 'oracle_policy_divergence': 0.9633202310651541}`
- `{'row_index': 3730, 'task': 'multi_needle', 'budget': 0.3, 'num_needles': 2, 'segment_start': 1698, 'segment_end': 1714, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9842389822006226, 'oracle_label': 0.03618840128183365, 'oracle_delta': -0.21257257461547852, 'retention_entropy': 0.11692722656157124, 'oracle_policy_divergence': 0.9480505809187889}`
- `{'row_index': 3908, 'task': 'multi_needle', 'budget': 0.2, 'num_needles': 2, 'segment_start': 1698, 'segment_end': 1714, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9842389822006226, 'oracle_label': 0.03618840128183365, 'oracle_delta': -0.21257257461547852, 'retention_entropy': 0.11692722656157124, 'oracle_policy_divergence': 0.9480505809187889}`
- `{'row_index': 4086, 'task': 'multi_needle', 'budget': 0.1, 'num_needles': 2, 'segment_start': 1698, 'segment_end': 1714, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9842389822006226, 'oracle_label': 0.03618840128183365, 'oracle_delta': -0.21257257461547852, 'retention_entropy': 0.11692722656157124, 'oracle_policy_divergence': 0.9480505809187889}`
- `{'row_index': 4264, 'task': 'multi_needle', 'budget': 0.05, 'num_needles': 2, 'segment_start': 1698, 'segment_end': 1714, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9842389822006226, 'oracle_label': 0.03618840128183365, 'oracle_delta': -0.21257257461547852, 'retention_entropy': 0.11692722656157124, 'oracle_policy_divergence': 0.9480505809187889}`
- `{'row_index': 4445, 'task': 'multi_needle', 'budget': 0.3, 'num_needles': 2, 'segment_start': 1746, 'segment_end': 1754, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9898019433021545, 'oracle_label': 0.04868219420313835, 'oracle_delta': -0.18780279159545898, 'retention_entropy': 0.08210328470047303, 'oracle_policy_divergence': 0.9411197490990162}`
- `{'row_index': 4623, 'task': 'multi_needle', 'budget': 0.2, 'num_needles': 2, 'segment_start': 1746, 'segment_end': 1754, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9898019433021545, 'oracle_label': 0.04868219420313835, 'oracle_delta': -0.18780279159545898, 'retention_entropy': 0.08210328470047303, 'oracle_policy_divergence': 0.9411197490990162}`
- `{'row_index': 4801, 'task': 'multi_needle', 'budget': 0.1, 'num_needles': 2, 'segment_start': 1746, 'segment_end': 1754, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9898019433021545, 'oracle_label': 0.04868219420313835, 'oracle_delta': -0.18780279159545898, 'retention_entropy': 0.08210328470047303, 'oracle_policy_divergence': 0.9411197490990162}`
- `{'row_index': 4979, 'task': 'multi_needle', 'budget': 0.05, 'num_needles': 2, 'segment_start': 1746, 'segment_end': 1754, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9898019433021545, 'oracle_label': 0.04868219420313835, 'oracle_delta': -0.18780279159545898, 'retention_entropy': 0.08210328470047303, 'oracle_policy_divergence': 0.9411197490990162}`
- `{'row_index': 5159, 'task': 'multi_needle', 'budget': 0.3, 'num_needles': 4, 'segment_start': 1780, 'segment_end': 1788, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9893856048583984, 'oracle_label': 0.05664337798953056, 'oracle_delta': -0.17501354217529297, 'retention_entropy': 0.08483921689808378, 'oracle_policy_divergence': 0.9327422268688679}`
- `{'row_index': 5339, 'task': 'multi_needle', 'budget': 0.2, 'num_needles': 4, 'segment_start': 1780, 'segment_end': 1788, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9893856048583984, 'oracle_label': 0.05664337798953056, 'oracle_delta': -0.17501354217529297, 'retention_entropy': 0.08483921689808378, 'oracle_policy_divergence': 0.9327422268688679}`
- `{'row_index': 5519, 'task': 'multi_needle', 'budget': 0.1, 'num_needles': 4, 'segment_start': 1780, 'segment_end': 1788, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9893856048583984, 'oracle_label': 0.05664337798953056, 'oracle_delta': -0.17501354217529297, 'retention_entropy': 0.08483921689808378, 'oracle_policy_divergence': 0.9327422268688679}`
- `{'row_index': 5699, 'task': 'multi_needle', 'budget': 0.05, 'num_needles': 4, 'segment_start': 1780, 'segment_end': 1788, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9893856048583984, 'oracle_label': 0.05664337798953056, 'oracle_delta': -0.17501354217529297, 'retention_entropy': 0.08483921689808378, 'oracle_policy_divergence': 0.9327422268688679}`
- `{'row_index': 5157, 'task': 'multi_needle', 'budget': 0.3, 'num_needles': 4, 'segment_start': 1748, 'segment_end': 1763, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9935708045959473, 'oracle_label': 0.07788945734500885, 'oracle_delta': -0.14770996570587158, 'retention_entropy': 0.056057398590045764, 'oracle_policy_divergence': 0.9156813472509384}`
- `{'row_index': 5337, 'task': 'multi_needle', 'budget': 0.2, 'num_needles': 4, 'segment_start': 1748, 'segment_end': 1763, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9935708045959473, 'oracle_label': 0.07788945734500885, 'oracle_delta': -0.14770996570587158, 'retention_entropy': 0.056057398590045764, 'oracle_policy_divergence': 0.9156813472509384}`
- `{'row_index': 5517, 'task': 'multi_needle', 'budget': 0.1, 'num_needles': 4, 'segment_start': 1748, 'segment_end': 1763, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9935708045959473, 'oracle_label': 0.07788945734500885, 'oracle_delta': -0.14770996570587158, 'retention_entropy': 0.056057398590045764, 'oracle_policy_divergence': 0.9156813472509384}`
- `{'row_index': 5697, 'task': 'multi_needle', 'budget': 0.05, 'num_needles': 4, 'segment_start': 1748, 'segment_end': 1763, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9935708045959473, 'oracle_label': 0.07788945734500885, 'oracle_delta': -0.14770996570587158, 'retention_entropy': 0.056057398590045764, 'oracle_policy_divergence': 0.9156813472509384}`
- `{'row_index': 3700, 'task': 'multi_needle', 'budget': 0.3, 'num_needles': 2, 'segment_start': 1420, 'segment_end': 1435, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9813248515129089, 'oracle_label': 0.07916032522916794, 'oracle_delta': -0.14630484580993652, 'retention_entropy': 0.13393584163659156, 'oracle_policy_divergence': 0.902164526283741}`
- `{'row_index': 3878, 'task': 'multi_needle', 'budget': 0.2, 'num_needles': 2, 'segment_start': 1420, 'segment_end': 1435, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9813248515129089, 'oracle_label': 0.07916032522916794, 'oracle_delta': -0.14630484580993652, 'retention_entropy': 0.13393584163659156, 'oracle_policy_divergence': 0.902164526283741}`
- `{'row_index': 4056, 'task': 'multi_needle', 'budget': 0.1, 'num_needles': 2, 'segment_start': 1420, 'segment_end': 1435, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9813248515129089, 'oracle_label': 0.07916032522916794, 'oracle_delta': -0.14630484580993652, 'retention_entropy': 0.13393584163659156, 'oracle_policy_divergence': 0.902164526283741}`
- `{'row_index': 4234, 'task': 'multi_needle', 'budget': 0.05, 'num_needles': 2, 'segment_start': 1420, 'segment_end': 1435, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9813248515129089, 'oracle_label': 0.07916032522916794, 'oracle_delta': -0.14630484580993652, 'retention_entropy': 0.13393584163659156, 'oracle_policy_divergence': 0.902164526283741}`
- `{'row_index': 5048, 'task': 'multi_needle', 'budget': 0.3, 'num_needles': 4, 'segment_start': 639, 'segment_end': 643, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9815876483917236, 'oracle_label': 0.08588554710149765, 'oracle_delta': -0.13919520378112793, 'retention_entropy': 0.1324310848160578, 'oracle_policy_divergence': 0.895702101290226}`
- `{'row_index': 5228, 'task': 'multi_needle', 'budget': 0.2, 'num_needles': 4, 'segment_start': 639, 'segment_end': 643, 'label_reason': 'oracle_drop', 'decision_type': 'over_keep', 'heuristic_keep_prob': 0.9815876483917236, 'oracle_label': 0.08588554710149765, 'oracle_delta': -0.13919520378112793, 'retention_entropy': 0.1324310848160578, 'oracle_policy_divergence': 0.895702101290226}`
