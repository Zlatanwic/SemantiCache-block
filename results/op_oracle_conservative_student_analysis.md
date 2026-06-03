# OP-SieveKV Oracle Dataset Analysis

- Dataset: `results/op_oracle_dataset_conservative.pt`
- Rows: 7896
- Oracle rows: 1076
- Policy checkpoint: `results/op_policy_oracle_conservative.pt`
- Policy probability field: `learned_keep_prob`
- Entropy threshold: 0.7931
- Divergence threshold: 0.2269

## KV-TIP Quadrants

- not_oracle: 6820
- Q4_uncertain_wrong: 204
- Q1_confident_aligned: 473
- Q2_uncertain_aligned: 334
- Q3_confident_wrong: 65

## Decision Types

- not_oracle: 6820
- aligned_keep: 301
- over_keep: 215
- aligned_drop: 509
- missed_keep: 51

## Oracle By Budget

### Budget 0.05

- Rows: 269
- Hard labels: 127
- Label mass: 124.8046
- Mean oracle delta: 0.3235
- Mean oracle label: 0.4992
- Reasons: `{'oracle_keep': 88, 'oracle_protected_drop': 40, 'oracle_mixed_drop': 129, 'oracle_soft_drop': 12}`

### Budget 0.1

- Rows: 269
- Hard labels: 136
- Label mass: 134.4116
- Mean oracle delta: 0.3235
- Mean oracle label: 0.4992
- Reasons: `{'oracle_keep': 88, 'oracle_protected_drop': 37, 'oracle_mixed_drop': 121, 'oracle_soft_drop': 23}`

### Budget 0.2

- Rows: 269
- Hard labels: 139
- Label mass: 148.5399
- Mean oracle delta: 0.3235
- Mean oracle label: 0.4992
- Reasons: `{'oracle_keep': 88, 'oracle_soft_drop': 25, 'oracle_mixed_drop': 124, 'oracle_protected_drop': 32}`

### Budget 0.3

- Rows: 269
- Hard labels: 128
- Label mass: 149.4997
- Mean oracle delta: 0.3235
- Mean oracle label: 0.4992
- Reasons: `{'oracle_keep': 88, 'oracle_soft_drop': 17, 'oracle_mixed_drop': 138, 'oracle_protected_drop': 26}`

## Top Q3 Confident-Wrong Rows

- `{'row_index': 1154, 'task': 'niah', 'budget': 0.05, 'needle_position': 0.25, 'segment_start': 337, 'segment_end': 353, 'label_reason': 'oracle_keep', 'decision_type': 'missed_keep', 'heuristic_keep_prob': 0.7177245020866394, 'learned_keep_prob': 0.22076520323753357, 'analysis_policy_prob': 0.22076520323753357, 'oracle_label': 0.7020658850669861, 'oracle_delta': 0.11857238411903381, 'retention_entropy': 0.7615622822293724, 'oracle_policy_divergence': 0.4813006818294525}`
- `{'row_index': 4946, 'task': 'multi_needle', 'budget': 0.05, 'num_needles': 2, 'segment_start': 1420, 'segment_end': 1435, 'label_reason': 'oracle_keep', 'decision_type': 'missed_keep', 'heuristic_keep_prob': 0.9813272356987, 'learned_keep_prob': 0.2379518300294876, 'analysis_policy_prob': 0.2379518300294876, 'oracle_label': 0.6443967819213867, 'oracle_delta': 0.09755992889404297, 'retention_entropy': 0.7916176217313704, 'oracle_policy_divergence': 0.4064449518918991}`
- `{'row_index': 3278, 'task': 'niah', 'budget': 0.05, 'needle_position': 1.0, 'segment_start': 169, 'segment_end': 184, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.7883561253547668, 'learned_keep_prob': 0.10254847258329391, 'analysis_policy_prob': 0.10254847258329391, 'oracle_label': 0.46756964921951294, 'oracle_delta': 0.0396076962351799, 'retention_entropy': 0.477022393671212, 'oracle_policy_divergence': 0.365021176636219}`
- `{'row_index': 2561, 'task': 'niah', 'budget': 0.05, 'needle_position': 0.75, 'segment_start': 82, 'segment_end': 105, 'label_reason': 'oracle_keep', 'decision_type': 'missed_keep', 'heuristic_keep_prob': 0.8100350499153137, 'learned_keep_prob': 0.14387960731983185, 'analysis_policy_prob': 0.14387960731983185, 'oracle_label': 0.507284939289093, 'oracle_delta': 0.05233134329319, 'retention_entropy': 0.5943096644216185, 'oracle_policy_divergence': 0.36340533196926117}`
- `{'row_index': 3285, 'task': 'niah', 'budget': 0.05, 'needle_position': 1.0, 'segment_start': 270, 'segment_end': 286, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9228777289390564, 'learned_keep_prob': 0.10242951661348343, 'analysis_policy_prob': 0.10242951661348343, 'oracle_label': 0.4544023871421814, 'oracle_delta': 0.03536810725927353, 'retention_entropy': 0.4766500066815105, 'oracle_policy_divergence': 0.35197287052869797}`
- `{'row_index': 2562, 'task': 'niah', 'budget': 0.05, 'needle_position': 0.75, 'segment_start': 105, 'segment_end': 119, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.7864365577697754, 'learned_keep_prob': 0.12250300496816635, 'analysis_policy_prob': 0.12250300496816635, 'oracle_label': 0.4744020104408264, 'oracle_delta': 0.041801467537879944, 'retention_entropy': 0.5365131339575527, 'oracle_policy_divergence': 0.35189900547266006}`
- `{'row_index': 4891, 'task': 'multi_needle', 'budget': 0.05, 'num_needles': 2, 'segment_start': 847, 'segment_end': 857, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9384012222290039, 'learned_keep_prob': 0.0835457444190979, 'analysis_policy_prob': 0.0835457444190979, 'oracle_label': 0.42291074991226196, 'oracle_delta': 0.025133132934570312, 'retention_entropy': 0.41455124620055905, 'oracle_policy_divergence': 0.33936500549316406}`
- `{'row_index': 3218, 'task': 'niah', 'budget': 0.1, 'needle_position': 1.0, 'segment_start': 169, 'segment_end': 184, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.7883561253547668, 'learned_keep_prob': 0.1337551474571228, 'analysis_policy_prob': 0.1337551474571228, 'oracle_label': 0.46756964921951294, 'oracle_delta': 0.0396076962351799, 'retention_entropy': 0.5676474804580325, 'oracle_policy_divergence': 0.33381450176239014}`
- `{'row_index': 3274, 'task': 'niah', 'budget': 0.05, 'needle_position': 1.0, 'segment_start': 105, 'segment_end': 119, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.7701447606086731, 'learned_keep_prob': 0.1357721984386444, 'analysis_policy_prob': 0.1357721984386444, 'oracle_label': 0.46772006154060364, 'oracle_delta': 0.03965603560209274, 'retention_entropy': 0.5730585741599481, 'oracle_policy_divergence': 0.33194786310195923}`
- `{'row_index': 191, 'task': 'niah', 'budget': 0.05, 'needle_position': 0.0, 'segment_start': 168, 'segment_end': 189, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.8282759189605713, 'learned_keep_prob': 0.07855592668056488, 'analysis_policy_prob': 0.07855592668056488, 'oracle_label': 0.39558184146881104, 'oracle_delta': 0.01608731411397457, 'retention_entropy': 0.3970703567549265, 'oracle_policy_divergence': 0.31702591478824615}`
- `{'row_index': 1850, 'task': 'niah', 'budget': 0.05, 'needle_position': 0.5, 'segment_start': 105, 'segment_end': 119, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.7679634690284729, 'learned_keep_prob': 0.12329187989234924, 'analysis_policy_prob': 0.12329187989234924, 'oracle_label': 0.43777894973754883, 'oracle_delta': 0.029985517263412476, 'retention_entropy': 0.5387498255703229, 'oracle_policy_divergence': 0.3144870698451996}`
- `{'row_index': 1137, 'task': 'niah', 'budget': 0.05, 'needle_position': 0.25, 'segment_start': 82, 'segment_end': 105, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.7957174777984619, 'learned_keep_prob': 0.10731921344995499, 'analysis_policy_prob': 0.10731921344995499, 'oracle_label': 0.41960519552230835, 'oracle_delta': 0.024048462510108948, 'retention_entropy': 0.4917765626114946, 'oracle_policy_divergence': 0.31228598207235336}`
- `{'row_index': 7152, 'task': 'multi_needle', 'budget': 0.05, 'num_needles': 8, 'segment_start': 1800, 'segment_end': 1816, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9838902354240417, 'learned_keep_prob': 0.14752371609210968, 'analysis_policy_prob': 0.14752371609210968, 'oracle_label': 0.45961278676986694, 'oracle_delta': 0.0370478630065918, 'retention_entropy': 0.6036085523448288, 'oracle_policy_divergence': 0.31208907067775726}`
- `{'row_index': 4713, 'task': 'multi_needle', 'budget': 0.1, 'num_needles': 2, 'segment_start': 847, 'segment_end': 857, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9384012222290039, 'learned_keep_prob': 0.11275095492601395, 'analysis_policy_prob': 0.11275095492601395, 'oracle_label': 0.42291074991226196, 'oracle_delta': 0.025133132934570312, 'retention_entropy': 0.5081583099950863, 'oracle_policy_divergence': 0.310159794986248}`
- `{'row_index': 2502, 'task': 'niah', 'budget': 0.1, 'needle_position': 0.75, 'segment_start': 105, 'segment_end': 119, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.7864365577697754, 'learned_keep_prob': 0.16796129941940308, 'analysis_policy_prob': 0.16796129941940308, 'oracle_label': 0.4744020104408264, 'oracle_delta': 0.041801467537879944, 'retention_entropy': 0.6530197787390539, 'oracle_policy_divergence': 0.30644071102142334}`
- `{'row_index': 3225, 'task': 'niah', 'budget': 0.1, 'needle_position': 1.0, 'segment_start': 270, 'segment_end': 286, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9228777289390564, 'learned_keep_prob': 0.15396685898303986, 'analysis_policy_prob': 0.15396685898303986, 'oracle_label': 0.4544023871421814, 'oracle_delta': 0.03536810725927353, 'retention_entropy': 0.6196789799915517, 'oracle_policy_divergence': 0.30043552815914154}`
- `{'row_index': 6389, 'task': 'multi_needle', 'budget': 0.05, 'num_needles': 4, 'segment_start': 1480, 'segment_end': 1486, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9784076809883118, 'learned_keep_prob': 0.13174790143966675, 'analysis_policy_prob': 0.13174790143966675, 'oracle_label': 0.4302245080471039, 'oracle_delta': 0.02752518653869629, 'retention_entropy': 0.5622123994111061, 'oracle_policy_divergence': 0.29847660660743713}`
- `{'row_index': 2501, 'task': 'niah', 'budget': 0.1, 'needle_position': 0.75, 'segment_start': 82, 'segment_end': 105, 'label_reason': 'oracle_keep', 'decision_type': 'missed_keep', 'heuristic_keep_prob': 0.8100350499153137, 'learned_keep_prob': 0.22087444365024567, 'analysis_policy_prob': 0.22087444365024567, 'oracle_label': 0.507284939289093, 'oracle_delta': 0.05233134329319, 'retention_entropy': 0.7617610000763059, 'oracle_policy_divergence': 0.28641049563884735}`
- `{'row_index': 202, 'task': 'niah', 'budget': 0.05, 'needle_position': 0.0, 'segment_start': 337, 'segment_end': 353, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.7880424857139587, 'learned_keep_prob': 0.11447417736053467, 'analysis_policy_prob': 0.11447417736053467, 'oracle_label': 0.40020036697387695, 'oracle_delta': 0.017629576846957207, 'retention_entropy': 0.5132656459609575, 'oracle_policy_divergence': 0.2857261896133423}`
- `{'row_index': 3777, 'task': 'multi_needle', 'budget': 0.2, 'num_needles': 2, 'segment_start': 388, 'segment_end': 405, 'label_reason': 'oracle_keep', 'decision_type': 'aligned_keep', 'heuristic_keep_prob': 0.9816324710845947, 'learned_keep_prob': 0.7989875674247742, 'analysis_policy_prob': 0.7989875674247742, 'oracle_label': 0.5140956044197083, 'oracle_delta': 0.05451178550720215, 'retention_entropy': 0.7239483446487132, 'oracle_policy_divergence': 0.2848919630050659}`
- `{'row_index': 3046, 'task': 'niah', 'budget': 0.05, 'needle_position': 1.0, 'segment_start': 270, 'segment_end': 286, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.931377112865448, 'learned_keep_prob': 0.11462308466434479, 'analysis_policy_prob': 0.11462308466434479, 'oracle_label': 0.3956623673439026, 'oracle_delta': 0.016114255413413048, 'retention_entropy': 0.5137049899556603, 'oracle_policy_divergence': 0.2810392826795578}`
- `{'row_index': 2574, 'task': 'niah', 'budget': 0.05, 'needle_position': 0.75, 'segment_start': 286, 'segment_end': 297, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9542773962020874, 'learned_keep_prob': 0.15815390646457672, 'analysis_policy_prob': 0.15815390646457672, 'oracle_label': 0.43721628189086914, 'oracle_delta': 0.029802605509757996, 'retention_entropy': 0.6298747639745534, 'oracle_policy_divergence': 0.2790623754262924}`
- `{'row_index': 446, 'task': 'niah', 'budget': 0.05, 'needle_position': 0.0, 'segment_start': 401, 'segment_end': 414, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9475811719894409, 'learned_keep_prob': 0.11563326418399811, 'analysis_policy_prob': 0.11563326418399811, 'oracle_label': 0.3871486186981201, 'oracle_delta': 0.013254895806312561, 'retention_entropy': 0.5166771725734681, 'oracle_policy_divergence': 0.271515354514122}`
- `{'row_index': 3214, 'task': 'niah', 'budget': 0.1, 'needle_position': 1.0, 'segment_start': 105, 'segment_end': 119, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.7701447606086731, 'learned_keep_prob': 0.19725708663463593, 'analysis_policy_prob': 0.19725708663463593, 'oracle_label': 0.46772006154060364, 'oracle_delta': 0.03965603560209274, 'retention_entropy': 0.7164082315426858, 'oracle_policy_divergence': 0.2704629749059677}`
- `{'row_index': 199, 'task': 'niah', 'budget': 0.05, 'needle_position': 0.0, 'segment_start': 290, 'segment_end': 306, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9460737705230713, 'learned_keep_prob': 0.1137508675456047, 'analysis_policy_prob': 0.1137508675456047, 'oracle_label': 0.3836405575275421, 'oracle_delta': 0.012070069089531898, 'retention_entropy': 0.5111270584557167, 'oracle_policy_divergence': 0.2698896899819374}`
- `{'row_index': 5742, 'task': 'multi_needle', 'budget': 0.3, 'num_needles': 4, 'segment_start': 373, 'segment_end': 388, 'label_reason': 'oracle_keep', 'decision_type': 'aligned_keep', 'heuristic_keep_prob': 0.9804646968841553, 'learned_keep_prob': 0.8021612167358398, 'analysis_policy_prob': 0.8021612167358398, 'oracle_label': 0.5325278043746948, 'oracle_delta': 0.06042361259460449, 'retention_entropy': 0.7175845459934144, 'oracle_policy_divergence': 0.269633412361145}`
- `{'row_index': 4949, 'task': 'multi_needle', 'budget': 0.05, 'num_needles': 2, 'segment_start': 1446, 'segment_end': 1452, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9794296026229858, 'learned_keep_prob': 0.1708759367465973, 'analysis_policy_prob': 0.1708759367465973, 'oracle_label': 0.4393211007118225, 'oracle_delta': 0.030486583709716797, 'retention_entropy': 0.6597046339821523, 'oracle_policy_divergence': 0.2684451639652252}`
- `{'row_index': 2362, 'task': 'niah', 'budget': 0.05, 'needle_position': 0.75, 'segment_start': 713, 'segment_end': 730, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.961828887462616, 'learned_keep_prob': 0.12170878797769547, 'analysis_policy_prob': 0.12170878797769547, 'oracle_label': 0.38921067118644714, 'oracle_delta': 0.013949491083621979, 'retention_entropy': 0.5342528586810471, 'oracle_policy_divergence': 0.2675018832087517}`
- `{'row_index': 132, 'task': 'niah', 'budget': 0.1, 'needle_position': 0.0, 'segment_start': 168, 'segment_end': 189, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.8282759189605713, 'learned_keep_prob': 0.12919577956199646, 'analysis_policy_prob': 0.12919577956199646, 'oracle_label': 0.39558184146881104, 'oracle_delta': 0.01608731411397457, 'retention_entropy': 0.5552284751024675, 'oracle_policy_divergence': 0.2663860619068146}`
- `{'row_index': 440, 'task': 'niah', 'budget': 0.05, 'needle_position': 0.0, 'segment_start': 306, 'segment_end': 317, 'label_reason': 'oracle_mixed_drop', 'decision_type': 'aligned_drop', 'heuristic_keep_prob': 0.9493993520736694, 'learned_keep_prob': 0.1745915561914444, 'analysis_policy_prob': 0.1745915561914444, 'oracle_label': 0.4404796063899994, 'oracle_delta': 0.030862733721733093, 'retention_entropy': 0.6681012962131592, 'oracle_policy_divergence': 0.265888050198555}`
