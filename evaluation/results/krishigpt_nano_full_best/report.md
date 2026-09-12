# KrishiGPT-nano evaluation - krishigpt_nano_full/best.pt

- val loss: **5.7076**  (perplexity **301.1**)
- generation throughput: **38 tok/s** (2560 tokens over 40 samples)

## Repetition statistics by decoding variant

| variant | distinct-1 | distinct-2 | rep-rate | mean max-run | frac rep>0.5 |
|---|---|---|---|---|---|
| greedy | 0.1 | 0.127 | 0.9143 | 31.2 | 1.0 |
| greedy_rep1.3 | 0.3406 | 0.4127 | 0.6698 | 37.8 | 0.8 |
| temp0.7 | 0.55 | 0.7619 | 0.4571 | 13.4 | 0.2 |
| temp0.9 | 0.8 | 0.9556 | 0.2032 | 2.4 | 0.0 |
| topk20 | 0.5469 | 0.854 | 0.4603 | 5.2 | 0.4 |
| topk40 | 0.6438 | 0.9429 | 0.3619 | 1.8 | 0.0 |
| topp0.9 | 0.8031 | 0.9746 | 0.2 | 2.2 | 0.0 |
| t0.9_k40_rep1.15 | 0.6937 | 0.9683 | 0.3111 | 1.0 | 0.0 |

## Samples (all prompts, all variants, no filtering)

### greedy
- **The best fertilizer for wheat** (max_tokens, d1=0.0625, rep=0.9523809523809523): The best fertilizer for wheat , and the same , and the same , and the same , and the same , and the same , and the same , and the same , and the same , and the same , and the same , and the same , and the same , and the same , and the same , and the same , and the same
- **Soil moisture affects** (max_tokens, d1=0.078125, rep=0.9365079365079365): Soil moisture affects of the soil . = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
- **To control pests in the field** (max_tokens, d1=0.125, rep=0.8888888888888888): To control pests in the field . The soil is a soil and the soil . The soil is a soil . The soil is a soil . The soil is a soil . = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
- **Rice is grown in** (max_tokens, d1=0.125, rep=0.8888888888888888): Rice is grown in the soil . The soil is a soil and the soil . The soil is a soil . The soil is a soil . The soil is a soil . = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
- **The farmer should irrigate** (max_tokens, d1=0.109375, rep=0.9047619047619048): The farmer should irrigate the soil . The soil is a soil . The soil is a soil . The soil is a soil . The soil is a soil . The soil is a soil . The soil is a soil . = = = = = = = = = = = = = = = = = = = = = = = = =

### greedy_rep1.3
- **The best fertilizer for wheat** (max_tokens, d1=0.828125, rep=0.1746031746031746): The best fertilizer for wheat , and the same is a great . [ Illustration : FIG . 22 . A LE ] " - T . S . _ _SAPER . CIODTY . BARGRING OF THE MROWUFHS . In 184 , the first of the
- **Soil moisture affects** (max_tokens, d1=0.078125, rep=0.9365079365079365): Soil moisture affects of the soil . = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
- **To control pests in the field** (max_tokens, d1=0.25, rep=0.7619047619047619): To control pests in the field . The soil is a plant and the soil , which are not be used to the soil . = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
- **Rice is grown in** (max_tokens, d1=0.21875, rep=0.7936507936507936): Rice is grown in the soil . The soil is a plant , and the soil can be used to the soil . = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
- **The farmer should irrigate** (max_tokens, d1=0.328125, rep=0.6825396825396826): The farmer should irrigate the soil . In the land , and the same is a large of the world , which are not be used to the plant . = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

### temp0.7
- **The best fertilizer for wheat** (max_tokens, d1=0.65625, rep=0.3492063492063492): The best fertilizer for wheat . They is a English holdings of the year the year and the body of the legumes . The labourer was even it was imford at the Paris , but you can be many of the harrow ' s of the Firfge of the Husbandman , and the first wiation of this price . In the ads of
- **Soil moisture affects** (max_tokens, d1=0.03125, rep=0.9841269841269841): Soil moisture affects . = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
- **To control pests in the field** (max_tokens, d1=0.703125, rep=0.30158730158730157): To control pests in the field . They are the land , is the year the plant and the body of the legumes . The soil was even it was imford at the Paris , but you can be many damage that harrow ' s will be taken for a natural plants . In the plant is a wiation of this price and the other rent of
- **Rice is grown in** (max_tokens, d1=0.6875, rep=0.31746031746031744): Rice is grown in the soil . Many land , is the year the plant and the body of the legumes . The soil was even it was imford at the Paris , but you can be many damage that harrow ' s will be taken for a natural plants . In the plant is a wiation of this price and the other rent of
- **The farmer should irrigate** (max_tokens, d1=0.671875, rep=0.3333333333333333): The farmer should irrigate the soil . Many land , is the year the year and the body of the legumes . The Middle century was it was imford at the Paris , but you can be many damage that harrow ' s , the Firfge of the Husbandman and plant . The world , the Mgrged the other rent of

### temp0.9
- **The best fertilizer for wheat** (max_tokens, d1=0.828125, rep=0.1746031746031746): The best fertilizer for wheat . They is a istyge . _the rowes and ealene. CONew CL Foit was imford . [ 90 ] Mprint of good cutting many damage that harrow ' s Tander . [ 6 ] Pm roots . ) , wirying of Mgrged , ¶ ads of
- **Soil moisture affects** (max_tokens, d1=0.703125, rep=0.30158730158730157): Soil moisture affects . They are the istyge of the leaves in the production . = = = = = = Foit was imford at rye - - cutting . = = = EnEquesharrow ' s Tander ) , a misesting of roots . Worose Phosphation of Mgrged the production of subsequ
- **To control pests in the field** (max_tokens, d1=0.828125, rep=0.1746031746031746): To control pests in the field . They is a istyge . _the amther and lene. CONew CL Foit was imford . [ 90 ] Mprint of good cutting many damage that harrow ' s Tander . [ 6 ] Pm roots . ) , wiation of Mgrged , ¶ ads of
- **Rice is grown in** (max_tokens, d1=0.8125, rep=0.19047619047619047): Rice is grown in statffowndlists structure . ¶ the amther production . ' s tothing being entirely Elial it was imford at rye and general cutting . = = = EnEquesharrow ' s Tander . In a misesting of roots . Worose wiation of Mgrged the other ads of
- **The farmer should irrigate** (max_tokens, d1=0.828125, rep=0.1746031746031746): The farmer should irrigate statffowndlists is the _the amther and lene. CONew CDFoit was imford . [ E- cutting . ) . ] See also quesharrow ' s Tander . [ 6 ] Pm roots . [ 5 ] Uns_ Mgrged , ¶ ads of

### topk20
- **The best fertilizer for wheat** (max_tokens, d1=0.484375, rep=0.5238095238095238): The best fertilizer for wheat . They is the land , is the year the year and the United Ages , for the tree , the same and was the time of which is used in a time of a same , that he was to take the year . The same plants of the same plant is to an best of this , and the other land and
- **Soil moisture affects** (max_tokens, d1=0.609375, rep=0.3968253968253968): Soil moisture affects . They are the land , is the year the plant and the food is , for the same , the same and the same time . The whole - plant should haue more of many crop that he can be used in the soil . Sargum and plant is a very most of this food . In the land and
- **To control pests in the field** (max_tokens, d1=0.578125, rep=0.42857142857142855): To control pests in the field . They is a plant , is the year the plant and the food . It is no reretion , it was not also . The plants is the soil or more of a large - field . The other use of agricultural acre , it may be used . The first plant , the plant has used the other land and
- **Rice is grown in** (max_tokens, d1=0.609375, rep=0.3968253968253968): Rice is grown in the soil . These has a soil are a small in the soil . The soil is no resoil from the soil . This is a most plants . The soil should be a plant in agricultural water and to have found with many crops , it may be used . The first soil , soil was the food and other land on
- **The farmer should irrigate** (max_tokens, d1=0.453125, rep=0.5555555555555556): The farmer should irrigate . " - - - - - - - - - - - - - - - - - - - - - T . S . " - - - BP_ of 188 - 3 - 7 , 4 ; a good , 321 . Hyt_ . Coon , and the Cyl

### topk40
- **The best fertilizer for wheat** (max_tokens, d1=0.578125, rep=0.42857142857142855): The best fertilizer for wheat . They is the land , is the year the year and the United Ages , for the tree , the same and was the time of which is used in a time of many years , that he ' s will be taken for a inputs of the roots . The first other , the common price is the other land and
- **Soil moisture affects** (max_tokens, d1=0.6875, rep=0.31746031746031744): Soil moisture affects . They are the land , is the year the plant and the farming ' s is no reretion , it was not also at most more used to produce the plant of many matter that he ' s will be a reds . Pisance , it is an irrigation of this has used the other parts of
- **To control pests in the field** (max_tokens, d1=0.625, rep=0.38095238095238093): To control pests in the field . They is a plant , is the year the plant and the farming ' s is no reretion , it was not also at most more used to produce the plant of many matter that he ' s will be a reds . Pelent plant is a very the soil to the number of the land and
- **Rice is grown in** (max_tokens, d1=0.625, rep=0.38095238095238093): Rice is grown in the soil . Many land , is the year the year and the United Ages , for the following agriculture was was it was the land . The whole - plant in the first of many crop that he ' s will be taken for a inputs of the roots . The first other , the common price and the other land on
- **The farmer should irrigate** (max_tokens, d1=0.703125, rep=0.30158730158730157): The farmer should irrigate . They do had been a soil . They were in the world . ' s . S . , i . 24 . [ 2 ] The common plant in the first of many of 203 ' s Tander . [ 6 ] Pisance , " - Prach Mees , __E

### topp0.9
- **The best fertilizer for wheat** (max_tokens, d1=0.703125, rep=0.30158730158730157): The best fertilizer for wheat . They is the land , is the year the oxygen and the body ' s tothing being entirely the same it was imford at rye and general cutting . = = = Enote queske ' s Tander . At a misesting of roots , it is wirying of this price and the other rent of
- **Soil moisture affects** (max_tokens, d1=0.828125, rep=0.1746031746031746): Soil moisture affects . They are the istyge till they the amther and body ' s tothing being entirely the same it was imford at rye and general cutting . = = No many damage that harrow ' s Arander have fge farmers used to roots . Worell wirying soil to gradtive production to re
- **To control pests in the field** (max_tokens, d1=0.78125, rep=0.2222222222222222): To control pests in the field . They is a English holdings is till they the oxygen and production . ' s tothing being entirely plants , it was imford at rye and general cutting . = = = Enote queske ' s Tander . In a misesting of roots . Worose wiation of Mgrged the other rent of
- **Rice is grown in** (max_tokens, d1=0.84375, rep=0.15873015873015872): Rice is grown in fish . TIdlists is till _Camther and lene. CONew CL Foit was imford at rye - - cutting . = OF IEnE- From the tenant , they are many foures of the roots . Worose wirying of Mgrged the other rent of
- **The farmer should irrigate** (max_tokens, d1=0.859375, rep=0.14285714285714285): The farmer should irrigate produce with the population . A structure are transins in the production . ' s tothing being entirely reactions , it was imford at rye and photosynthesis . The picary of many damage that harrow ' s Arander have fge farmers used to roots . Worose wiation of this price is the other rent of

### t0.9_k40_rep1.15
- **The best fertilizer for wheat** (max_tokens, d1=0.703125, rep=0.30158730158730157): The best fertilizer for wheat . They is a plant , is the year the year and the United Ages , for the tree , while this can be not also at which is used to produce the first of many crop that he ' s will be taken on a inputs of the roots . ) , such as the common food is the other land and
- **Soil moisture affects** (max_tokens, d1=0.703125, rep=0.30158730158730157): Soil moisture affects . They are the land , is a year the plant and the food ' s of the tree , the same and they had also at which is used to produce the soil of many crop that he can be used in the seeds . Sarkbent plant is a very incous , and the other parts of
- **To control pests in the field** (max_tokens, d1=0.640625, rep=0.36507936507936506): To control pests in the field . They is a plant , is the year the plant and the food ' s of the tree , the same land was not also at which is used to produce the soil of many crop that he can be used in the seeds . Sesenm and plant is a very best soil to the number of the land on
- **Rice is grown in** (max_tokens, d1=0.671875, rep=0.3333333333333333): Rice is grown in the soil . Many land , is a year the plant and the food ' s of the acre , the same farmer was not also at which is used to produce the first of many crop that he ' s will be taken for a inputs of the roots . The first other , the common price is the other parts of
- **The farmer should irrigate** (max_tokens, d1=0.75, rep=0.25396825396825395): The farmer should irrigate . They are the land , is a year the plant and the food ' s time the tree , the same and was not also at which is used to produce the first of many crop that he ' s will be taken for a inputs of the roots . [ Illustration : FIG . 124 . In 1718 –
