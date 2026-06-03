# OP-SieveKV Oracle Dataset Analysis

- Dataset: `results/op_oracle_dataset_pilot.pt`
- Rows: 7896
- Oracle rows: 1076
- Policy checkpoint: `results/op_policy_oracle_pilot.pt`
- Policy probability field: `learned_keep_prob`
- Entropy threshold: 0.8973
- Divergence threshold: 0.0580

## KV-TIP Quadrants

- not_oracle: 6820
- Q2_uncertain_aligned: 382
- Q1_confident_aligned: 425
- Q4_uncertain_wrong: 156
- Q3_confident_wrong: 113

## Decision Types

- not_oracle: 6820
- missed_keep: 49
- aligned_keep: 303
- aligned_drop: 717
- over_keep: 7

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

- `{'row_index': 4946, 'task': 'multi_needle', 'budget': 0.05, 'num_needles': 2, 'segment_start': 1420, 'segment_end': 1435, 'label_reason': 'oracle_keep', 'decision_type': 'missed_keep', 'heuristic_keep_prob': 0.9813272356987, 'learned_keep_prob': 0.28297290205955505, 'analysis_policy_prob': 0.28297290205955505, 'oracle_label': 0.6443967819213867, 'oracle_delta': 0.09755992889404297, 'retention_entropy': 0.859470042017252, 'oracle_policy_divergence': 0.36142387986183167}`
- `{'row_index': 4412, 'task': 'multi_needle', 'budget': 0.3, 'num_needles': 2, 'segment_start': 1420, 'segment_end': 1435, 'label_reason': 'oracle_keep', 'decision_type': 'missed_keep', 'heuristic_keep_prob': 0.9813272356987, 'learned_keep_prob': 0.29732611775398254, 'analysis_policy_prob': 0.29732611775398254, 'oracle_label': 0.6443967819213867, 'oracle_delta': 0.09755992889404297, 'retention_entropy': 0.8779977649465925, 'oracle_policy_divergence': 0.3470706641674042}`
- `{'row_index': 4768, 'task': 'multi_needle', 'budget': 0.1, 'num_needles': 2, 'segment_start': 1420, 'segment_end': 1435, 'label_reason': 'oracle_keep', 'decision_type': 'missed_keep', 'heuristic_keep_prob': 0.9813272356987, 'learned_keep_prob': 0.31154847145080566, 'analysis_policy_prob': 0.31154847145080566, 'oracle_label': 0.6443967819213867, 'oracle_delta': 0.09755992889404297, 'retention_entropy': 0.8949528238272241, 'oracle_policy_divergence': 0.33284831047058105}`
- `{'row_index': 4891, 'task': 'multi_needle', 'budget': 0.05, 'num_needles': 2, 'segment_start': 847, 'segment_end': 857, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9384012222290039, 'learned_keep_prob': 0.11407078057527542, 'analysis_policy_prob': 0.11407078057527542, 'oracle_label': 0.42291074991226196, 'oracle_delta': 0.025133132934570312, 'retention_entropy': 0.5120738562703914, 'oracle_policy_divergence': 0.30883996933698654}`
- `{'row_index': 4713, 'task': 'multi_needle', 'budget': 0.1, 'num_needles': 2, 'segment_start': 847, 'segment_end': 857, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9384012222290039, 'learned_keep_prob': 0.13911467790603638, 'analysis_policy_prob': 0.13911467790603638, 'oracle_label': 0.42291074991226196, 'oracle_delta': 0.025133132934570312, 'retention_entropy': 0.581915528671908, 'oracle_policy_divergence': 0.2837960720062256}`
- `{'row_index': 2561, 'task': 'niah', 'budget': 0.05, 'needle_position': 0.75, 'segment_start': 82, 'segment_end': 105, 'label_reason': 'oracle_keep', 'decision_type': 'missed_keep', 'heuristic_keep_prob': 0.8100350499153137, 'learned_keep_prob': 0.27000972628593445, 'analysis_policy_prob': 0.27000972628593445, 'oracle_label': 0.507284939289093, 'oracle_delta': 0.05233134329319, 'retention_entropy': 0.84147859247007, 'oracle_policy_divergence': 0.23727521300315857}`
- `{'row_index': 4056, 'task': 'multi_needle', 'budget': 0.1, 'num_needles': 2, 'segment_start': 1420, 'segment_end': 1435, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9813248515129089, 'learned_keep_prob': 0.3099632263183594, 'analysis_policy_prob': 0.3099632263183594, 'oracle_label': 0.07916032522916794, 'oracle_delta': -0.14630484580993652, 'retention_entropy': 0.8931310049215402, 'oracle_policy_divergence': 0.23080290108919144}`
- `{'row_index': 3730, 'task': 'multi_needle', 'budget': 0.3, 'num_needles': 2, 'segment_start': 1698, 'segment_end': 1714, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9842389822006226, 'learned_keep_prob': 0.26536741852760315, 'analysis_policy_prob': 0.26536741852760315, 'oracle_label': 0.03618840128183365, 'oracle_delta': -0.21257257461547852, 'retention_entropy': 0.8347383449630008, 'oracle_policy_divergence': 0.2291790172457695}`
- `{'row_index': 3645, 'task': 'multi_needle', 'budget': 0.3, 'num_needles': 2, 'segment_start': 847, 'segment_end': 857, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9383347034454346, 'learned_keep_prob': 0.2645587623119354, 'analysis_policy_prob': 0.2645587623119354, 'oracle_label': 0.03756533935666084, 'oracle_delta': -0.2094707489013672, 'retention_entropy': 0.833547982167816, 'oracle_policy_divergence': 0.22699342295527458}`
- `{'row_index': 3700, 'task': 'multi_needle', 'budget': 0.3, 'num_needles': 2, 'segment_start': 1420, 'segment_end': 1435, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9813248515129089, 'learned_keep_prob': 0.30428948998451233, 'analysis_policy_prob': 0.30428948998451233, 'oracle_label': 0.07916032522916794, 'oracle_delta': -0.14630484580993652, 'retention_entropy': 0.8864713069615935, 'oracle_policy_divergence': 0.2251291647553444}`
- `{'row_index': 4535, 'task': 'multi_needle', 'budget': 0.2, 'num_needles': 2, 'segment_start': 847, 'segment_end': 857, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9384012222290039, 'learned_keep_prob': 0.19903258979320526, 'analysis_policy_prob': 0.19903258979320526, 'oracle_label': 0.42291074991226196, 'oracle_delta': 0.025133132934570312, 'retention_entropy': 0.7199890500047674, 'oracle_policy_divergence': 0.2238781601190567}`
- `{'row_index': 5157, 'task': 'multi_needle', 'budget': 0.3, 'num_needles': 4, 'segment_start': 1748, 'segment_end': 1763, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9935708045959473, 'learned_keep_prob': 0.29647204279899597, 'analysis_policy_prob': 0.29647204279899597, 'oracle_label': 0.07788945734500885, 'oracle_delta': -0.14770996570587158, 'retention_entropy': 0.8769355010510744, 'oracle_policy_divergence': 0.21858258545398712}`
- `{'row_index': 6249, 'task': 'multi_needle', 'budget': 0.05, 'num_needles': 4, 'segment_start': 22, 'segment_end': 44, 'label_reason': 'oracle_keep', 'decision_type': 'aligned_keep', 'heuristic_keep_prob': 0.9786722660064697, 'learned_keep_prob': 0.6993536949157715, 'analysis_policy_prob': 0.6993536949157715, 'oracle_label': 0.9058898687362671, 'oracle_delta': 0.2311561107635498, 'retention_entropy': 0.8820795034256561, 'oracle_policy_divergence': 0.2065361738204956}`
- `{'row_index': 3908, 'task': 'multi_needle', 'budget': 0.2, 'num_needles': 2, 'segment_start': 1698, 'segment_end': 1714, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9842389822006226, 'learned_keep_prob': 0.2409193068742752, 'analysis_policy_prob': 0.2409193068742752, 'oracle_label': 0.03618840128183365, 'oracle_delta': -0.21257257461547852, 'retention_entropy': 0.7965657152020444, 'oracle_policy_divergence': 0.20473090559244156}`
- `{'row_index': 4234, 'task': 'multi_needle', 'budget': 0.05, 'num_needles': 2, 'segment_start': 1420, 'segment_end': 1435, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9813248515129089, 'learned_keep_prob': 0.28069859743118286, 'analysis_policy_prob': 0.28069859743118286, 'oracle_label': 0.07916032522916794, 'oracle_delta': -0.14630484580993652, 'retention_entropy': 0.8564009531472383, 'oracle_policy_divergence': 0.20153827220201492}`
- `{'row_index': 3595, 'task': 'multi_needle', 'budget': 0.3, 'num_needles': 2, 'segment_start': 342, 'segment_end': 354, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9424342513084412, 'learned_keep_prob': 0.23077823221683502, 'analysis_policy_prob': 0.23077823221683502, 'oracle_label': 0.02996879070997238, 'oracle_delta': -0.22817373275756836, 'retention_entropy': 0.7793654721676176, 'oracle_policy_divergence': 0.20080944150686264}`
- `{'row_index': 5337, 'task': 'multi_needle', 'budget': 0.2, 'num_needles': 4, 'segment_start': 1748, 'segment_end': 1763, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9935708045959473, 'learned_keep_prob': 0.278229296207428, 'analysis_policy_prob': 0.278229296207428, 'oracle_label': 0.07788945734500885, 'oracle_delta': -0.14770996570587158, 'retention_entropy': 0.8530268692579015, 'oracle_policy_divergence': 0.20033983886241913}`
- `{'row_index': 2501, 'task': 'niah', 'budget': 0.1, 'needle_position': 0.75, 'segment_start': 82, 'segment_end': 105, 'label_reason': 'oracle_keep', 'decision_type': 'missed_keep', 'heuristic_keep_prob': 0.8100350499153137, 'learned_keep_prob': 0.3077782094478607, 'analysis_policy_prob': 0.3077782094478607, 'oracle_label': 0.507284939289093, 'oracle_delta': 0.05233134329319, 'retention_entropy': 0.8905921138441958, 'oracle_policy_divergence': 0.1995067298412323}`
- `{'row_index': 5306, 'task': 'multi_needle', 'budget': 0.2, 'num_needles': 4, 'segment_start': 1454, 'segment_end': 1469, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9813317060470581, 'learned_keep_prob': 0.2954005300998688, 'analysis_policy_prob': 0.2954005300998688, 'oracle_label': 0.10705997049808502, 'oracle_delta': -0.1196904182434082, 'retention_entropy': 0.8755956601050181, 'oracle_policy_divergence': 0.18834055960178375}`
- `{'row_index': 7152, 'task': 'multi_needle', 'budget': 0.05, 'num_needles': 8, 'segment_start': 1800, 'segment_end': 1816, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9838902354240417, 'learned_keep_prob': 0.2729160785675049, 'analysis_policy_prob': 0.2729160785675049, 'oracle_label': 0.45961278676986694, 'oracle_delta': 0.0370478630065918, 'retention_entropy': 0.8456179743420043, 'oracle_policy_divergence': 0.18669670820236206}`
- `{'row_index': 6419, 'task': 'multi_needle', 'budget': 0.05, 'num_needles': 4, 'segment_start': 1780, 'segment_end': 1788, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9891647696495056, 'learned_keep_prob': 0.3057267963886261, 'analysis_policy_prob': 0.3057267963886261, 'oracle_label': 0.12242703139781952, 'oracle_delta': -0.10757160186767578, 'retention_entropy': 0.8881790416904396, 'oracle_policy_divergence': 0.18329976499080658}`
- `{'row_index': 5517, 'task': 'multi_needle', 'budget': 0.1, 'num_needles': 4, 'segment_start': 1748, 'segment_end': 1763, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9935708045959473, 'learned_keep_prob': 0.2602477967739105, 'analysis_policy_prob': 0.2602477967739105, 'oracle_label': 0.07788945734500885, 'oracle_delta': -0.14770996570587158, 'retention_entropy': 0.8271200710401736, 'oracle_policy_divergence': 0.18235833942890167}`
- `{'row_index': 5697, 'task': 'multi_needle', 'budget': 0.05, 'num_needles': 4, 'segment_start': 1748, 'segment_end': 1763, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9935708045959473, 'learned_keep_prob': 0.24985304474830627, 'analysis_policy_prob': 0.24985304474830627, 'oracle_label': 0.07788945734500885, 'oracle_delta': -0.14770996570587158, 'retention_entropy': 0.8110451228018045, 'oracle_policy_divergence': 0.17196358740329742}`
- `{'row_index': 4086, 'task': 'multi_needle', 'budget': 0.1, 'num_needles': 2, 'segment_start': 1698, 'segment_end': 1714, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9842389822006226, 'learned_keep_prob': 0.20651280879974365, 'analysis_policy_prob': 0.20651280879974365, 'oracle_label': 0.03618840128183365, 'oracle_delta': -0.21257257461547852, 'retention_entropy': 0.7347640100789349, 'oracle_policy_divergence': 0.17032440751791}`
- `{'row_index': 3285, 'task': 'niah', 'budget': 0.05, 'needle_position': 1.0, 'segment_start': 270, 'segment_end': 286, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9228777289390564, 'learned_keep_prob': 0.2857167720794678, 'analysis_policy_prob': 0.2857167720794678, 'oracle_label': 0.4544023871421814, 'oracle_delta': 0.03536810725927353, 'retention_entropy': 0.8631238553407684, 'oracle_policy_divergence': 0.16868561506271362}`
- `{'row_index': 5486, 'task': 'multi_needle', 'budget': 0.1, 'num_needles': 4, 'segment_start': 1454, 'segment_end': 1469, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9813317060470581, 'learned_keep_prob': 0.27555009722709656, 'analysis_policy_prob': 0.27555009722709656, 'oracle_label': 0.10705997049808502, 'oracle_delta': -0.1196904182434082, 'retention_entropy': 0.8493164220358524, 'oracle_policy_divergence': 0.16849012672901154}`
- `{'row_index': 7523, 'task': 'multi_needle', 'budget': 0.2, 'num_needles': 8, 'segment_start': 1848, 'segment_end': 1856, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9893843531608582, 'learned_keep_prob': 0.12791506946086884, 'analysis_policy_prob': 0.12791506946086884, 'oracle_label': 0.28936734795570374, 'oracle_delta': -0.02187669277191162, 'retention_entropy': 0.5516924034435253, 'oracle_policy_divergence': 0.1614522784948349}`
- `{'row_index': 3823, 'task': 'multi_needle', 'budget': 0.2, 'num_needles': 2, 'segment_start': 847, 'segment_end': 857, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9383347034454346, 'learned_keep_prob': 0.19834452867507935, 'analysis_policy_prob': 0.19834452867507935, 'oracle_label': 0.03756533935666084, 'oracle_delta': -0.2094707489013672, 'retention_entropy': 0.7186047708390564, 'oracle_policy_divergence': 0.1607791893184185}`
- `{'row_index': 4357, 'task': 'multi_needle', 'budget': 0.3, 'num_needles': 2, 'segment_start': 847, 'segment_end': 857, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9384012222290039, 'learned_keep_prob': 0.26566052436828613, 'analysis_policy_prob': 0.26566052436828613, 'oracle_label': 0.42291074991226196, 'oracle_delta': 0.025133132934570312, 'retention_entropy': 0.835168608900257, 'oracle_policy_divergence': 0.15725022554397583}`
- `{'row_index': 5666, 'task': 'multi_needle', 'budget': 0.05, 'num_needles': 4, 'segment_start': 1454, 'segment_end': 1469, 'label_reason': 'oracle_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9813317060470581, 'learned_keep_prob': 0.2593619227409363, 'analysis_policy_prob': 0.2593619227409363, 'oracle_label': 0.10705997049808502, 'oracle_delta': -0.1196904182434082, 'retention_entropy': 0.8257819779273334, 'oracle_policy_divergence': 0.15230195224285126}`
