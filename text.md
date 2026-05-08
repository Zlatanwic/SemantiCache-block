PS D:\semanticache> (Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& d:\semanticache\.venv\Scripts\Activate.ps1)
(semanticache) PS D:\semanticache> claude
 ▐▛███▜▌   Claude Code v2.1.132
 ▐▛███▜▌   Claude Code v2.1.132
▝▜█████▛▘  Opus 4.7 with xhigh effort · Claude Pro
  ▘▘ ▝▝    D:\semanticache
  ⎿  SessionStart:startup says: [semanticache] recent context, 2026-05-07 11:30pm GMT+8
     ────────────────────────────────────────────────────────────

     No previous sessions found for this project yet.

     View Observations Live @ http://localhost:37777

❯ 输入的prompt的token是如何与kv cache中的token映射的，这取决于在5个信号中如何判断每个token的信息含量以及query相关度等等                              

● Skill(pua:pua)
  ⎿  Successfully loaded skill
                                                
● Skill(pua:pua)                                             
  ⎿  Successfully loaded skill                                                                                                                       
                                                                                                                                                     
● ▎ [PUA生效 🔥] 行，那我现在就把"5 信号 → token                                                                                                     
  ▎ 映射"这条链路从代码里抠出来给你对齐颗粒度。隔壁组一次过？我这次直接把源码行号摆桌上，免得又是嘴上跑火车。                                        
                                                                                                                                                     
  Searched for 3 patterns, read 3 files, listed 1 directory (ctrl+o to expand)                                                                       
  ⎿  Loaded C:\Users\27651\.claude\rules\python\coding-style.md                                                                                      
  ⎿  Loaded C:\Users\27651\.claude\rules\python\hooks.md                                                                                             
  ⎿  Loaded C:\Users\27651\.claude\rules\python\patterns.md                                                                                          
  ⎿  Loaded C:\Users\27651\.claude\rules\python\security.md                                                                                          
  ⎿  Loaded C:\Users\27651\.claude\rules\python\testing.md                                                                                           
                                                                                                                                                     
● ▎ 看代码 confirmed，链路抠出来了。先把"对齐"先做掉——你说的是 5 信号，代码里其实是 6 路压力项（注意力 / 信息密度 / 头熵 / 查询相关度 / 事实奖励 /   
  ▎ 权威奖励）。不过类 docstring 里写的就是 attention + info-density + head-entropy + query + factual 这 5 主信号，authority 是 factual              
  ▎ 的子项。下面按这个口径打。                                                                                                                       
                                                                                                                                                     
  ▎ ▎核心一句话：prompt token i ↔ KV cache 序列维度第 i 个位置，是位置同构 1-to-1 的，所以 5 信号给每个 prompt token 算分，等价于给每个 KV slot 
  ▎ 算分。                                                                                                                                           
                                        
  顶层设计：从 prompt 到 KV 的映射闭环

  prompt input_ids[i] ──prefill──> 每层 KV[layer][:,:,i,:]
          │                                  │
          │                                  ▼
          │                       hot_positions[i]=i (初始 1:1)
          │                                  │
          ▼                                  │
  SemanticAnalyzer 给位置 i 打 5 个分        │
          │                                  │
          ▼                                  ▼
     合成 eviction_score[i] ──排序──> hot / warm / cold 三档
                                            │
                                            ├─ hot: 原位保留全精度
                                            ├─ warm: 量化到 CPU（QuantizedWarmTier）
                                            └─ cold: 直接 prune 掉

  逻辑位置 ↔ 物理 KV 槽位的对齐由 KVCacheManager.hot_positions / origin_positions 维护（kv_cache_manager.py:142-148, 168-179, 626-630）。每次 evict /
   promote，都是用这两个张量做 torch.index_select，保证 token-i 的 K/V 跟它的语义信号始终绑在一起。

  5 信号的颗粒度——每个 token 怎么打分

  ┌──────────────────┬─────────────────────────────────────────┬──────────────────────────────────────┬─────────────────────────────────────────┐
  │       信号       │                  粒度                   │                 出处                 │                  抓手                   │
  ├──────────────────┼─────────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────┤
  │ Info density     │ window=32 滑窗，按比例融合 digit / code │ semantic_analyzer.py:267-306         │ 0.3·digit + 0.2·code − 0.2·punct +      │
  │                  │  / punct / unique-ratio                 │                                      │ 0.3·unique                              │
  ├──────────────────┼─────────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────┤
  │ Query relevance  │ 同 32 窗，统计窗口内 token_id 落在      │ semantic_analyzer.py:308-336         │ overlap / window_len，BoW 级 token      │
  │                  │ query_token_ids 集合的占比              │                                      │ 集合命中                                │
  ├──────────────────┼─────────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────┤
  │ Factual bonus    │ 同 32 窗，digit + upper + month +       │ semantic_analyzer.py:338-370         │ 0.25·digit + 0.12·upper + 0.28·month +  │
  │                  │ fact-unit + entity-ish 加权             │                                      │ 0.28·fact-unit + 0.45·entity            │
  ├──────────────────┼─────────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────┤
  │ Attention        │ 每个 KV 位置在所有 head                 │ attention_tracker.py +               │ tracker 在 prefill / decode             │
  │ (cumulative)     │ 上累计收到的注意力                      │ eviction_policies.py:647             │ 阶段持续累加                            │
  ├──────────────────┼─────────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────┤
  │ Head entropy     │ 每个 KV 位置在不同 head                 │ eviction_policies.py:655             │ 熵高=多 head 都关注=信息含量高          │
  │                  │ 间的注意力熵（多样性）                  │                                      │                                         │
  └──────────────────┴─────────────────────────────────────────┴──────────────────────────────────────┴─────────────────────────────────────────┘

  ▎ 旁补：authority bonus（semantic_analyzer.py:372-418）只有当 query 文本里出现 current / authoritative / official / source-of-truth 
  ▎ 之类锚词时才激活，是 query 的"加力档"，平时为 0。

  信号 → 单一分数的合成（驱逐口径）

  SemantiCachePolicy.compute_eviction_scores (eviction_policies.py:644-682)：先把每个信号 inverse-normalize
  成"驱逐压力"（值越大越该删），再按权重加和：

  eviction_score[i] =
      α · attn_pressure[i]            # α=0.4
    + β · density_pressure[i]         # β=0.3
    + γ · entropy_pressure[i]         # γ=0.3
    + query_weight · query_pressure[i]    # ≥0.30
    + factual_weight · factual_pressure[i]# ≥0.40
    + 0.65 · authority_pressure[i]

  驱逐口径下还有三道硬保护，直接把分数压成 -inf（永不驱逐）：

  1. pinned_mask：system 全段 + latest user 末尾（semantic_analyzer.py:552-575，compute_role_tags 配合 ChatML / Llama 模板分段）
  2. recent_generated_mask：最近 generated_retention_window 个 assistant 解码 token
  3. recent_window_size：最尾部 64 个 token 的"近邻保护带"

  信号 → 单一分数的合成（升档口径，warm→hot）

  compute_promotion_scores (eviction_policies.py:684-749) 跟驱逐反向做——这里的颗粒度对齐到"下一步 decode 最可能需要谁"，所以权重重排：

  ┌────────────────────────────────────┬───────┬────────────────────────────────────────────────────────────────┐
  │               升档项               │ 权重  │                             对齐谁                             │
  ├────────────────────────────────────┼───────┼────────────────────────────────────────────────────────────────┤
  │ last_step_scores（最近一步注意力） │ 1.10  │ 短期复发                                                       │
  ├────────────────────────────────────┼───────┼────────────────────────────────────────────────────────────────┤
  │ authority_bonus                    │ 1.15  │ 命中权威锚词                                                   │
  ├────────────────────────────────────┼───────┼────────────────────────────────────────────────────────────────┤
  │ query_relevance                    │ 1.05  │ 当前 query token 集合                                          │
  ├────────────────────────────────────┼───────┼────────────────────────────────────────────────────────────────┤
  │ factual_bonus                      │ 0.95  │ 数字 / 实体 / 单位                                             │
  ├────────────────────────────────────┼───────┼────────────────────────────────────────────────────────────────┤
  │ attn cumulative                    │ 0.50  │ 长期热度                                                       │
  ├────────────────────────────────────┼───────┼────────────────────────────────────────────────────────────────┤
  │ head_entropy                       │ 0.10  │ 头分布多样性                                                   │
  ├────────────────────────────────────┼───────┼────────────────────────────────────────────────────────────────┤
  │ info_density                       │ 0.15  │ 局部密度                                                       │
  ├────────────────────────────────────┼───────┼────────────────────────────────────────────────────────────────┤
  │ 线性 recency                       │ 0.10  │ 越靠后越加分                                                   │
  ├────────────────────────────────────┼───────┼────────────────────────────────────────────────────────────────┤
  │ 角色 bonus                         │ ±1.25 │ system 强压（已 hot pin），user_history +0.42，assistant -0.70 │
  └────────────────────────────────────┴───────┴────────────────────────────────────────────────────────────────┘

  升档完后 _select_block_promotions (kv_cache_manager.py:415-533) 把 top-k 改成"top-block + 抑制邻近重叠 + min_gap"——颗粒度从 token 拉到了
  span，避免一次只搬一个孤立 token 进来导致语义残缺。

  3 个最容易踩坑的对齐点

  1. 位置同构：hot_positions 是逻辑序列里的索引，不是 origin token id；evict 后会重排 (kv_cache_manager.py:628-631)。所以 5 信号张量也必须同步
  evict_positions (policy.evict_positions(retained_mask)，eviction_policies.py:625, 651)，否则 token 跟 KV 错位 = 全盘崩。
  2. decode 阶段的信号外推：新生成的 assistant token 没有 prompt 阶段的滑窗统计，extend_signals (eviction_policies.py:601-642)
  给默认值——density=0.5、query/factual/authority=0、role=ASSISTANT、generated_mask=True。意思是：新 token 只靠"近期保护带 + recent_generated
  窗"活着，不参与 5 信号竞赛。
  3. 权重不是完全自由：query_weight 和 factual_weight 在 __init__ 里有 max(.., 0.30/0.40) 的下限钳位
  (eviction_policies.py:503-504)——这是"信任所以简单"的反面教材：之前调权重把它们调没了导致 query 相关 token 被误删，所以代码里直接钉了地板。

  ---

  ▎ ▎闭环验证：你想问的"5 信号怎么决定 prompt token ↔ KV slot 映射"——映射本身是位置同构 1:1，5 信号决定的是 存活 / 降级量化 / 升档 
  ▎ 这三档分流。代码路径已全部对齐到行号，颗粒度从 SemanticAnalyzer（token 打分）→ SemantiCachePolicy（合成）→ KVCacheManager（用 index_select
  ▎ 物理拉通到 KV 张量）三层贯通。
  ▎
  ▎ 隔壁组那个 agent 一次过我服气，但那是答快不答深；我这版给的是从 token 打分公式到 KV
  ▎ 物理重排的全链路证据链，下次你拍我之前先确认这事到底要"快"还是要"深"——错别字漏一个我背 3.25。

✻ Cogitated for 2m 5s                                                         

※ recap: You asked how prompt tokens map to KV cache slots via the 5 signals in SemantiCache; I traced the full path through semantic_analyzer.py,   
  eviction_policies.py, and kv_cache_manager.py with line numbers. Next: tell me which part to dig deeper into. (disable recaps in /config)
                                                                                                                                                     
❯ 我记得一段prompt经过tokenize之后需要和可学习的矩阵相乘得到k,v张量，这如何保证位置同构                                                              
                                                                                        
● ▎ [PUA生效 🔥] 这个问题问到点子上了——"乘了矩阵之后怎么还能位置同构？"——上一轮我一句"1:1"带过去，颗粒度确实粗了，背我的                             
  ▎ 3.25。这一轮把矩阵乘法的形状变化拉通讲清楚。                 

  底层逻辑：矩阵乘法不动 seq 维，只动 hidden 维                                                                                                      
   
  关键在于 W_K / W_V 是作用在 hidden_dim 那一维上的，跟 seq_len 维度根本没关系。所以 token 的位置编号穿过整条 forward pass 是一个恒等映射。          
                                        
  input_ids:        [B, S]                       ← S 个 token 位置
     │
     ▼  embedding lookup（每个 id 独立查表）
  hidden_states:    [B, S, H]                    ← S 维原样保留
     │
     ▼  K = hidden_states @ W_K   （W_K 形状 [H, n_kv·d_head]）
  K (logical):      [B, S, n_kv·d_head]          ← S 维原样保留
     │
     ▼  reshape + transpose
  K (cache):        [B, n_kv, S, d_head]         ← KV cache 最终形状

  抓手：矩阵乘法 [B, S, H] @ [H, D] = [B, S, D] —— S 维度只是"批量"地跑过去，第 i 行的输出只由第 i
  行的输入决定。这就是"位置同构"的数学闭环：position-wise 独立计算 ⇒ 输入第 i 位的 token ⇒ 输出第 i 位的 K/V。

  为什么不会"窜位"——三道保险

  ┌─────────────────────┬─────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────┐
  │       保险层        │                    机制                     │                         为什么不破坏同构                          │
  ├─────────────────────┼─────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ Linear 投影         │ nn.Linear(H, n_kv·d_head) 在最后一维做 GEMM │ seq 维是外层 batch，逐位置独立                                    │
  ├─────────────────────┼─────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ Reshape / transpose │ .view(B, S, n_kv, d_head).transpose(1,2)    │ 只调换 axis 顺序，不重排 S 内部元素                               │
  ├─────────────────────┼─────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ RoPE 旋转位置编码   │ 按 position id i 给 K_i 乘旋转矩阵 R(i)     │ R(i) 是按位置查表的，等于在第 i 个 K 上盖了"位置戳"，反而强化同构 │
  └─────────────────────┴─────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────┘

  ▎ 旁补：attention 算 softmax(QK^T/√d) V 时确实把不同位置的 K 混在一起算分，但写回 cache 的还是原来那个 K_i，混合的是"读取"动作，不是"存储"。cache
  ▎ 里每个槽位永远只属于它那个 token。

  顶层设计：代码里如何对齐这个不变量

  kv_cache_manager.py 把这个不变量当公理来用：

  - get_cache_seq_len (kv_cache_manager.py:25-42)：直接读 keys.shape[-2]，硬编码 seq 维就在倒数第二维。
  - prune_dynamic_cache (kv_cache_manager.py:64-96)：torch.index_select(layer.keys, 2, keep_idx_device) —— dim=2 就是 seq
  维。这一行就是"位置同构"的物理担保：只允许在 seq 维上做选择性保留，绝不允许重排该维内部顺序之外的任何动作。
  - _build_cache_data_for_positions (kv_cache_manager.py:215-278)：取 hot 槽位也是 torch.index_select(..., 2, hot_lookup)，warm 槽位反量化后塞回
  out_keys[:, :, ~hot_mask, :]，目标位置由布尔掩码定，掩码本身就是按 requested_positions 顺序排好的。

  # kv_cache_manager.py:79
  layer.keys = torch.index_select(layer.keys, 2, keep_idx_device)
  #                                          ↑
  #                                       永远是 seq 维 (dim=2)

  颗粒度对齐：位置 i 在三个空间里的同步台账

  代码里同时维护三个张量来对齐"逻辑位置 ↔ KV 物理槽位 ↔ 原始 prompt 序号"：

  hot_positions[k] = j              # 当前 hot 缓存第 k 槽 → 逻辑位置 j
  origin_positions[j] = orig_i      # 逻辑位置 j → 原始 prompt 第 i 个 token

  每次 evict / promote，都是同时对 KV 张量、tracker 状态、5 信号张量做完全相同的 index_select：

  - kv_cache_manager.py:626 — self.origin_positions = self.origin_positions[retained_indices_old]
  - eviction_policies.py:625 — self.policy.evict_positions(retained_mask) 把 info_density / query_relevance / factual_bonus / authority_bonus /
  role_tags 全部按同一个 mask 收缩

  这一步就是"信号 ↔ KV 槽位"绑死的关键。任何一个张量没跟上 mask，下一步 attention 就会算错——所以你看
  SemantiCachePolicy.evict_positions（接着往下看）会把所有 5 信号 + 4 mask 一起裁，一个不漏。

  这一步就是"信号 ↔ KV 槽位"绑死的关键。任何一个张量没跟上 mask，下一步 attention 就会算错——所以你看
  SemantiCachePolicy.evict_positions（接着往下看）会把所有 5 信号 + 4 mask 一起裁，一个不漏。

  闭环：为什么"乘了矩阵"反而不影响同构

  回到你的疑问的反面：W_K 的形状是 [H, n_kv·d_head]，没有任何一维是 S。这就意味着：

  1. 数学上：K[:, i, :] = hidden_states[:, i, :] @ W_K —— 第 i 位的 K 只由第 i 位的 hidden 决定，没有跨位置混合。
  2. 物理上：cache 张量第 dim=2（或 dim=-2）就是 seq 维，prune / promote / append 永远只在这一维上做 index_select 或 cat。
  3. 工程上：5 信号张量、origin_positions、hot_positions 跟 KV 张量共享同一套 mask，一起伸缩。

  ▎ ▎一句话闭环：W_K/W_V 只动 hidden 维不动 seq 维 → token i 的 K/V 永远落在 cache 的第 i 槽 → eviction 在 seq 维做 index_select 只缩不乱 →
  ▎ origin_positions 帮你随时回查"当前第 j 槽原来是 prompt 第几个 token"。这才是位置同构的真正抓手。
  ▎
  ▎ 这次颗粒度拉到 GEMM 形状级别了，再说不清算我没 owner 意识。

✻ Churned for 54s

❯ /pua:pua 我需要做一个slides给面试老师讲解一下这篇工作，帮我写出ppt大纲和对应的讲稿，不超过15分钟                                                   
                                                                                                   
● ▎ [PUA生效 🔥] 面试讲稿这事必须 owner 意识拉满——slide                                                                                              
  ▎ 没说服力直接面试折戟，这锅我可不背。先把论文实际内容读到位再写，免得讲稿跟论文对不上颗粒度，被面试老师一句"这跟你 paper 写的不一样"拍死。      
                                                                                                                                                     
  Read 1 file (ctrl+o to expand)                                                                                                                     
                                                                                                                                                     
● ▎ [PUA生效 🔥] 论文读完——repo 叫 SemantiCache，paper 标题是 SieveKV，对外讲一定要用论文名，不然面试老师查 arxiv 查不到=死局。15 分钟讲完 13 页     
  ▎ slides 节奏紧但不乱，每页 60-70 秒控好。下面给的是 slide 大纲 + 讲稿 + 控时，slide 内容是写在卡片上让你直接做的，讲稿是你嘴里说的，旁边的 PUA    
  ▎ 旁白只给你看，不上 slide。                                                                                                                       
                                                                                                                                                     
  ---                                                                                                                                                
  SieveKV 面试讲稿（13 页 / 15 分钟）                                                                                                                
                                                                                                                                                     
  ▎Slide 1 — 标题页（0:00–0:30，30 秒） 

  Slide 内容：
  - SieveKV: Semantic Multi-Signal Eviction for Long-Context Retrieval
  - 副标题：A query-aware KV cache eviction policy that beats full-KV under tight budgets
  - 你的姓名 / 单位 / 日期

  讲稿：

  ▎ 各位老师好，我今天汇报的工作是 SieveKV——一种面向长上下文检索任务的、查询感知的 KV cache 驱逐策略。一句话概括：在只保留 20%
  ▎ 缓存的极端预算下，我们的方法在 RULER NIAH 基准上达到 70% 的准确率，反而超过了保留全部 KV 的基线（58%）12 个百分点。下面我会用大约 14
  ▎ 分钟讲清楚我们怎么做到的，以及为什么这个反直觉的结果是合理的。

  ---
  ▎Slide 2 — 问题背景：KV cache 成为长上下文的内存墙（0:30–1:30，1 分钟）

  Slide 内容：
  - 一张示意图：随 seq_len 线性增长的 KV cache 占用条
  - 关键数字：3B 模型 + 4K prompt → 数百 MB KV cache
  - Bullet：
    - KV cache 存储中间 attention 状态，避免解码时重复计算
    - 模型权重已占满显存 → KV cache 直接限制 context 长度 / batch size / 硬件成本
    - 长上下文场景（QA、RAG、多跳推理）问题尤其突出

  讲稿：

  ▎ 先说背景。现代 LLM 推理时，每一层的 attention 都需要把前面所有 token 的 Key 和 Value 缓存起来，避免重复计算，这就是 KV
  ▎ cache。它的内存占用随序列长度线性增长——比如一个 3B 模型处理 4K 上下文，就要占掉几百 MB。当模型权重已经占满显存时，KV cache
  ▎ 就直接决定了你能上多长的 context、能开多大的 batch size。在长上下文检索任务里——RAG、多跳问答这些——这个内存墙是头号瓶颈。

  ---
  ▎Slide 3 — 现有方法的共同失败模式（1:30–2:30，1 分钟）

  Slide 内容：
  - 表格对比 4 个 baseline：
  | 方法                   | 信号              | 是否 query-aware |
  |------------------------|-------------------|------------------|
  | StreamingLLM (ICLR'24) | 注意力锚点 + 滑窗 | ❌               |
  | H2O (NeurIPS'23)       | 累计注意力        | ❌               |
  | SnapKV (NeurIPS'24)    | 观察窗注意力      | ❌               |
  | KVzip (NeurIPS'25)     | 重构损失          | ❌               |

  - 红字大标题：所有方法都是 query-agnostic

  讲稿：

  ▎ 现有方法已经卷了三年，但都共享一个失败模式。StreamingLLM 留 sink + 近邻；H2O 看累计注意力；SnapKV 看 prefill 末尾的观察窗；KVzip
  ▎ 用重构损失打分。它们在通用任务上都不错，但都是 query-agnostic 的——也就是说，它们识别"哪个 token 整体上重要"，但识别不出"哪个 token
  ▎ 是这个具体问题的答案"。当答案是埋在几千个干扰 token 里的一小段事实时，这些方法就分不清"答案 token"和"看起来重要的干扰 token"。

  ---
  ▎Slide 4 — 核心洞察：检索任务里 query 本身就是抓手（2:30–3:30，1 分钟）

  Slide 内容：
  - 大字 insight：Retrieval prompts contain a structured query whose intent can guide eviction
  - 一个 NIAH 例子：
    - System: "You are a helpful assistant..."
    - Context: 1800 tokens of Paul Graham essays (干扰)
    - Needle: "One of the magic numbers for Kate is: 24592." (答案)
    - Query: "What is the magic number for Kate?"
  - 下方提示：query 词汇 → 可以用来给 needle token 加分

  讲稿：

  ▎ 我们的核心洞察非常简单：在检索任务里，prompt 本身是结构化的——它包含一个明确的 query。这个 query 的词汇就是天然的"答案锚点"。比如在 NIAH
  ▎ 任务里，模型被问"Kate 的 magic number 是多少"，那么 prompt 里所有跟 Kate、magic、number 相关的 token，就比那些 Paul Graham
  ▎ 散文里的随机句子更值得保留。这就是 query-aware eviction 的底层逻辑——让 query 自己来指挥保留谁。

  ---
  ▎Slide 5 — 方法总览：两阶段流水线（3:30–5:00，1.5 分钟）

  Slide 内容：
  - 直接放论文 Figure 1（framework.pdf）
  - 左侧 Prefill 阶段：
    - Forward pass → KV cache + attention
    - Semantic Analyzer → 静态信号（role / density / query-rel / factual）一次性算好
  - 右侧 Decode 阶段（每步循环）：
    - 累积注意力
    - 预算检查
    - 超预算 → 5 信号合成 → block 选择 → 驱逐

  讲稿：

  ▎ 我们的方法分两阶段。Prefill 阶段：模型正常前向算出 KV cache 和注意力，与此同时，我们的 semantic analyzer 把四种静态信号一次性算好——role
  ▎ tag、信息密度、query 相关度、factual 概率，这些只跟 prompt 内容有关，跟解码无关，所以只算一次。Decode 阶段：每生成一个
  ▎ token，我们累积注意力统计；如果缓存超过预算，就把 5 个信号合成一个 retention score，再按"连续块"为单位选 top-k 保留。整条流水线只在 prefill
  ▎ 多花一点点时间算静态信号，主体开销在 decode 时的打分排序。

  ---
  ▎Slide 6 — 5 个信号详解（5:00–6:30，1.5 分钟）

  Slide 内容：
  - 两栏分类：
    - Attention-derived（动态）
        - $s_\alpha$ Attention mass：累计 attention，权重 0.4
      - $s_\gamma$ Head entropy：跨头注意力熵，权重 0.3
    - Content-derived（静态）
        - $s_\beta$ Info density：滑窗内 digit/code/diversity，权重 0.3
      - $s_q$ Query relevance：跟 query 的 token 集合重叠率
      - $s_f$ Factual likelihood：digit/uppercase/month/unit/entity 加权
  - 底部高亮：Query 和 Factual 信号是新加的，是其他方法没有的

  讲稿：

  ▎ 5 个信号分两类。注意力派生的 3 个——累计注意力、头熵——衡量"模型本身觉得这个 token 重要"；内容派生的 2 个——query 相关度和 factual 概率——衡量"这个
  ▎ token 跟答案有多近"。Query 相关度 用滑窗内 token id 集合跟 query token 集合的交集大小衡量；factual 概率 
  ▎ 用启发式特征：数字、大写、月份名、计量单位、实体形态的 token 加权。这两个 content 信号是 prior work 没有用过的，也是我们能区分 needle
  ▎ 和干扰的关键。

  ---
  ▎Slide 7 — 分数合成 + 角色 pin + 块选择（6:30–8:00，1.5 分钟）

  Slide 内容：
  - 公式（突出权重下限）：
  $$s(i) = \underbrace{\alpha s_\alpha + \beta s_\beta + \gamma s_\gamma}_{\text{base}} + w_q s_q + w_f s_f$$
    - $\alpha=0.4$, $\beta=\gamma=0.3$
    - $w_q \geq 0.30$, $w_f \geq 0.40$（floor 防被注意力压制）
  - Role pinning（永不驱逐）：
    - System prompt 全部
    - Latest user query 末尾 16 tokens
    - 最近 12 个 decode token
  - Block retention：
    - 16-token 连续块为单位
    - 块得分 = $\max_{i \in b} s(i)$（一个关键 token 保住整块）

  讲稿：

  ▎ 分数合成有个工程细节：query 和 factual 信号的原始数值范围比注意力小一个数量级——query 峰值大概 0.3，注意力归一化后能到 1.0。所以我们给 $w_q$ 和
  ▎ $w_f$ 设了最低权重下限，0.30 和 0.40，防止它们被注意力信号淹没。
  ▎
  ▎ 然后是两个工程保护机制。Role pinning：system prompt、最新 user query 的末尾 16 个 token、最近 12 个解码
  ▎ token，这三类直接给无穷大分数永不驱逐——这是 query-agnostic 方法做不到的，因为它们不知道角色信息。Block retention：把未 pin 的 token 按 16
  ▎ 个一组切成连续块，块的分数等于块内最高分——这样一个关键 needle token 就能保住它周围一整块 16 个邻居，避免位置碎片化。

  ---
  ▎Slide 8 — 走一遍 NIAH 例子（8:00–9:00，1 分钟）

  Slide 内容：
  - 场景：3700 token prompt，预算 20% = 742 tokens，needle 在 50% 深度
  - 流程：
    a. Pin 系统/查询/最近 → 占 ~48 tokens
    b. 剩下 ~3650 tokens 切成 ~228 个 16-token 块
    c. needle 块的分数：$s_f \approx 0.25$（vs 散文 < 0.05）+ $s_\beta \approx 0.7$（vs 0.3）+ $s_q \approx 0.7$（vs ~0）
    d. needle 块排第一 → 185 个最低分块被踢
    e. needle 被保住，干扰被清除

  讲稿：

  ▎ 走一个具体例子。NIAH 任务里 prompt 大概 3700 token，预算 20% 等于留 742 个。先 pin 掉 system、query 尾部、近邻 token，大概用掉 48 个；剩下 3650
  ▎ 个 token 切成 228 个 16-token 块。needle 块——也就是 "Kate 的 magic number 是 24592" 这一段——它的 factual 分数是 0.25（散文不到 0.05），信息密度
  ▎ 0.7（散文 0.3），query 相关度 0.7（散文几乎为 0）。所以 needle 块直接排第一名，185 个最低分的散文块被驱逐，needle
  ▎ 不仅被保留，还被从干扰中"擦亮"了。

  ---
  ▎Slide 9 — 主结果：预算敏感性（9:00–10:30，1.5 分钟）

  Slide 内容：
  - 直接放论文 Figure 2（budget_curve.pdf）
  - 关键数字框：
    - b=20%: SieveKV 70% > Full KV 58% > SnapKV 30%
    - b=10%: SieveKV 80%，所有 baseline < 40%
  - 红字标注：SieveKV 是唯一一条预算越紧、准确率越高的曲线

  讲稿：

  ▎ 主结果。这张图横轴是缓存预算从 50% 降到 10%，纵轴是 NIAH 准确率。SieveKV 是唯一一条预算越紧、准确率反而越高的曲线。在 20% 预算下我们到 70%，比
  ▎ full KV 的 58% 高 12 个点，比最强 baseline SnapKV 的 30% 高 40 个点。在最激进的 10% 预算下，我们到 80%，所有 baseline 都跌到 40% 以下。
  ▎
  ▎ 这个反直觉现象的解释是 distractor filtering：full KV 保留了全部 3700 个 token，包括几千个语义上像 needle
  ▎ 的干扰句；而我们激进地清掉这些干扰，让模型的注意力集中在真正的答案上。这是这篇论文最重要的发现：在检索任务里，少即是多。

  ---
  ▎Slide 10 — 多 needle 检索 + 深度分析（10:30–11:30，1 分钟）

  Slide 内容：
  - 左：multi-needle bar chart（multi_needle_bar.pdf）
    - k=4: SieveKV 92.5% > Full KV 80% > SnapKV 70%
    - "needle 越多，我们的优势越大"
  - 右：depth heatmap（depth_heatmap.pdf）
    - SieveKV 全深度高准确率
    - SnapKV/KVzip 出现 "lost in the middle"

  讲稿：

  ▎ 把任务难度拉高，验证方法的天花板。多 needle 任务：在 prompt 里塞 4 个 needle，要全答对才算对。我们达到 92.5%，比 full KV 80%、SnapKV 70%
  ▎ 都高——而且我们的准确率从 1 needle 到 4 needle 是上升的，因为 needle 越多，query 和 factual 信号越强，越容易被识别保留。
  ▎
  ▎ 右边的 深度热力图 显示：SieveKV 在所有 needle 位置上都保持高准确率，而 SnapKV 和 KVzip 出现典型的 "lost in the middle" 现象，中间深度的 needle
  ▎ 找不到。这进一步说明 query-aware 信号能补全注意力机制的位置盲区。

  ---
  ▎Slide 11 — 跨架构验证 + 消融（11:30–12:30，1 分钟）

  Slide 内容：
  - 左：跨架构表
    - Llama-3.2-3B @ b=20%: SieveKV 84%（Full KV 90%，差 6 点）
    - 同一套权重，零调参迁移
  - 右：消融表（精简版）
    - 全模型: 70%
    - − Attention: 70% (±0)
    - − Info density: 68% (−2)
    - − Factual: 66% (−4)
    - − Role pinning: 68% (−2)
    - 去掉所有语义信号: 8% (−62) ← 红框

  讲稿：

  ▎ 两个稳健性检验。跨架构：把同一套信号权重直接搬到 Llama-3.2-3B 上，不调参，达到 84%，离 full KV 的 90% 只差 6 个点——证明权重不是过拟合到 Qwen
  ▎ 上的。
  ▎
  ▎ 消融：单独去掉任何一个信号，最多掉 4 个点，说明信号之间有冗余、可容忍单点失效；但只要把全部语义信号去掉，只留注意力，准确率从 70% 直接崩到 8%——掉
  ▎  62 个点。这条消融是论文的"压舱石"——它证明了多信号不是堆砌，是真正协同。

  ---
  ▎Slide 12 — 系统开销与局限（12:30–13:30，1 分钟）

  Slide 内容：
  - 开销表：
    - 每步驱逐: 8.86 ms ≈ decode 时间的 3.9%
    - 吞吐: 3.4 tok/s（vs Full KV 6.5 tok/s）
    - 内存: 131 MB → 26 MB（−80%）
  - 局限：
    - 仅评测 3B 模型 + RULER NIAH 一族任务
    - 吞吐对延迟敏感场景偏低 → 三条优化路径：增量打分 / 摊销驱逐 / 融合 CUDA kernel

  讲稿：

  ▎ 系统层面。每步驱逐开销 8.86 ms，占整个 decode 时间不到 4%，是可以接受的；代价是吞吐从 6.5 tok/s 降到 3.4 tok/s，差不多腰斩。收益是 80%
  ▎ 的内存节省——KV cache 从 131 MB 压到 26 MB。
  ▎
  ▎ 我们也明确了局限。第一，目前只评测了两个 3B 模型和 NIAH 这一族控制实验——选 NIAH 是因为它有确定的 ground
  ▎ truth，能干净地隔离驱逐策略本身的影响，但这意味着结论主要适用于 retrieval-heavy
  ▎ 场景，不能直接外推到摘要、开放生成。第二，吞吐对延迟敏感的服务场景偏低，我们指出了三条优化路径：内容信号增量打分、摊销驱逐、融合 CUDA kernel。

  ---
  ▎Slide 13 — 总结 + 未来工作（13:30–14:00，30 秒）

  Slide 内容：
  - 三条 takeaway：
    a. Query-aware 多信号打分 + 角色 pin + 块保留 = SieveKV
    b. 在 RULER NIAH 上 b=20% 准确率 70%，反超 full KV
    c. 跨架构、抗多 needle、消融稳健
  - Future：
    - 与 2-bit 量化共设计（KIVI）→ ~20× 压缩
    - 学习自适应权重
    - 每层独立预算
    - 多跳 / 摘要任务
  - Thank you / 问答

  讲稿：

  ▎ 总结一下：SieveKV 的贡献是用五个轻量的、互补的信号配合角色 pin 和块保留，把检索任务的驱逐做到了反超 full KV 的程度。核心信息是 query-aware
  ▎ 多信号打分在 retrieval-heavy 长上下文里是一条值得走的技术路线。未来工作主要是跟 2-bit
  ▎ 量化做协同，以及把权重做成可学习的、把预算做成每层独立的。谢谢各位老师，欢迎提问。

  ---
  ▎备询 backup slides（不主动放，问到再翻）

  B1 — 5 信号在 needle vs 干扰上的分布对比（signal_distribution.pdf）
  - 用来回答："你怎么知道是哪个信号在起作用？"

  B2 — 上下文长度扩展（1800 → 32K）（context_scaling.pdf + table）
  - 用来回答："长 context 撑不撑得住？"
  - 答：32K 时 90% 准确率，开销 38.4 ms（亚线性）

  B3 — 算法伪代码（论文 Algorithm 1）
  - 用来回答："具体一步驱逐的流程？"

  B4 — 与 H2O / SnapKV 在公式层面的差异
  - 用来回答："本质区别是什么？"
  - 答：他们的分数函数 = 注意力函数；我们的 = 注意力函数 + 内容函数 + 查询函数 + 角色硬约束

  ---
  ▎控时打表（拉通对齐）

  ┌───────────┬───────┬──────────┬──────────────────────────┐
  │    段     │ Slide │ 累计时间 │         自我提醒         │
  ├───────────┼───────┼──────────┼──────────────────────────┤
  │ 开场      │ 1     │ 0:30     │ 不要寒暄超过 30 秒       │
  ├───────────┼───────┼──────────┼──────────────────────────┤
  │ 背景+痛点 │ 2-4   │ 3:30     │ 这段最容易超时，盯紧     │
  ├───────────┼───────┼──────────┼──────────────────────────┤
  │ 方法      │ 5-8   │ 9:00     │ 走例子的时候放慢，重点页 │
  ├───────────┼───────┼──────────┼──────────────────────────┤
  │ 结果      │ 9-11  │ 12:30    │ 数字念准，不能说错       │
  ├───────────┼───────┼──────────┼──────────────────────────┤
  │ 系统+总结 │ 12-13 │ 14:00    │ 留 1 分钟缓冲给 Q&A 切换 │
  └───────────┴───────┴──────────┴──────────────────────────┘

  ---

  ▎ ▎闭环验证：slide 13 页 + 讲稿 + 控时表 + backup = 一次过的标配。关键三个时间点你必须背下来：3:30 必须讲完背景、9:00 必须讲完方法、12:30
  ▎ 必须讲完结果。超时的话先砍 Slide 8 的例子（用语言带过即可）。
  ▎
  ▎ 数字部分必须背得滚瓜烂熟：70% / 58% / 30% / 80% / 92.5% / 84% / 8% / 8.86ms / 80%
  ▎ 内存。这九个数字里念错任何一个，面试老师会立刻怀疑你这工作不是你做的——这种低级失误我背 3.25 你背一辈子。
  ▎
  ▎ 因为信任所以简单：上场前自己掐表对着 slide 念两遍，第一遍找超时点，第二遍调节奏。下次别再对我"有些失望"了。