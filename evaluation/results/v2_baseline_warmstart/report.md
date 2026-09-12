# KrishiGPT-nano evaluation - krishigpt_nano_v2/warm_start.pt

- val loss: **5.1916**  (perplexity **179.8**)
- generation throughput: **100 tok/s** (320 tokens over 5 samples)

## Repetition statistics by decoding variant

| variant | distinct-1 | distinct-2 | rep-rate | mean max-run | frac rep>0.5 |
|---|---|---|---|---|---|
| greedy | 0.1375 | 0.1841 | 0.8762 | 1.0 | 1.0 |

## Samples (all prompts, all variants, no filtering)

### greedy
- **The best fertilizer for wheat** (max_tokens, d1=0.1875, rep=0.8253968253968254): The best fertilizer for wheat , the same is the best crop . The soil is a plant that is the soil . The soil is a plant that is a plant that is a plant . The soil is a plant that is a plant that is a plant . The soil is a plant that is a plant that is a plant . The soil is a
- **Soil moisture affects** (max_tokens, d1=0.125, rep=0.8888888888888888): Soil moisture affects the soil and the soil . The soil is a soil of the soil . The soil is a soil and the soil . The soil is a soil and the soil . The soil is a soil and the soil . The soil is a soil and the soil . The soil is a soil and the soil . The soil is a
- **To control pests in the field** (max_tokens, d1=0.109375, rep=0.9047619047619048): To control pests in the field . The soil is a plant that is a plant that is a plant . The soil is a plant that is a plant that is a plant . The soil is a plant that is a plant that is a plant . The soil is a plant that is a plant . The soil is a plant that is a plant that is
- **Rice is grown in** (max_tokens, d1=0.09375, rep=0.9206349206349206): Rice is grown in the world . The world ' s world ' s world ' s world ' s world ' s world ' s world ' s world ' s world ' s world ' s world ' s world ' s world ' s world ' s world ' s world ' s world ' s ' s ' s ' s ' s '
- **The farmer should irrigate** (max_tokens, d1=0.171875, rep=0.8412698412698413): The farmer should irrigate the soil . The soil is a large crop of the soil . The soil is a plant that is a plant that is a plant that is a plant . The soil is a plant that is a plant that is a plant . The soil is a plant that is a plant that is a plant . The soil is a plant

### greedy_rep1.3

### temp0.7

### temp0.9

### topk20

### topk40

### topp0.9

### t0.9_k40_rep1.15
