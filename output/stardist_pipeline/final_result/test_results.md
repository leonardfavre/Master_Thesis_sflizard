# Test results

## Segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 24734 |
| TP | 173754 |
| FN | 23301 |
| Precision | 0.8753879327717544 |
| Recall | 0.8817538250742178 |
| Accuracy | 0.7834202778316328 |
| F1 | 0.8785593475298513 |
| n_true | 197055 |
| n_pred | 198488 |
| mean_true_score | 0.7054480878163006 |
| mean_matched_score | 0.8000510661316638 |
| panoptic_quality | 0.7028923426511965 |
## Segmentation Metrics per class

| Metric | Neutrophil | Epithelial | Lymphocyte | Plasma | Eosinophil | Connective tissue |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| FP |2736.0000 |  12602.0000 |  10780.0000 |  13666.0000 |  2228.0000 |  10113.0000 | 
| TP |349.0000 |  64041.0000 |  44640.0000 |  10514.0000 |  1145.0000 |  25524.0000 | 
| FN |336.0000 |  20447.0000 |  13548.0000 |  3061.0000 |  597.0000 |  12892.0000 | 
| Precision |0.1131 |  0.8356 |  0.8055 |  0.4348 |  0.3395 |  0.7162 | 
| Recall |0.5095 |  0.7580 |  0.7672 |  0.7745 |  0.6573 |  0.6644 | 
| Accuracy |0.1020 |  0.6596 |  0.6473 |  0.3860 |  0.2884 |  0.5260 | 
| F1 |0.1851 |  0.7949 |  0.7859 |  0.5570 |  0.4477 |  0.6893 | 
| n_true |685.0000 |  84488.0000 |  58188.0000 |  13575.0000 |  1742.0000 |  38416.0000 | 
| n_pred |3085.0000 |  76643.0000 |  55420.0000 |  24180.0000 |  3373.0000 |  35637.0000 | 
| mean_true_score |0.3775 |  0.5799 |  0.6516 |  0.6469 |  0.4594 |  0.5385 | 
| mean_matched_score |0.7409 |  0.7651 |  0.8494 |  0.8352 |  0.6989 |  0.8105 | 
| panoptic_quality |0.1372 |  0.6082 |  0.6675 |  0.4652 |  0.3129 |  0.5587 | 
## Confusion Matrix

![](images/Confusion_Matrix.png)

## Confusion Matrix normalized

![](images/Confusion_Matrix_normalized.png)

## Segmentation Metrics per class after graph improvement

| Metric | Neutrophil | Epithelial | Lymphocyte | Plasma | Eosinophil | Connective tissue |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| FP |2041.0000 |  16872.0000 |  8590.0000 |  10520.0000 |  1330.0000 |  9554.0000 | 
| TP |301.0000 |  68520.0000 |  43664.0000 |  9385.0000 |  1010.0000 |  26551.0000 | 
| FN |384.0000 |  15968.0000 |  14524.0000 |  4190.0000 |  732.0000 |  11865.0000 | 
| Precision |0.1285 |  0.8024 |  0.8356 |  0.4715 |  0.4316 |  0.7354 | 
| Recall |0.4394 |  0.8110 |  0.7504 |  0.6913 |  0.5798 |  0.6911 | 
| Accuracy |0.1104 |  0.6760 |  0.6539 |  0.3895 |  0.3288 |  0.5535 | 
| F1 |0.1989 |  0.8067 |  0.7907 |  0.5606 |  0.4949 |  0.7126 | 
| n_true |685.0000 |  84488.0000 |  58188.0000 |  13575.0000 |  1742.0000 |  38416.0000 | 
| n_pred |2342.0000 |  85392.0000 |  52254.0000 |  19905.0000 |  2340.0000 |  36105.0000 | 
| mean_true_score |0.3281 |  0.6206 |  0.6388 |  0.5769 |  0.4022 |  0.5600 | 
| mean_matched_score |0.7466 |  0.7653 |  0.8513 |  0.8344 |  0.6937 |  0.8102 | 
| panoptic_quality |0.1485 |  0.6173 |  0.6731 |  0.4678 |  0.3433 |  0.5774 | 
## Confusion Matrix after graph improvement

![](images/Confusion_Matrix_after_graph_improvement.png)

## Confusion Matrix normalized after graph improvement

![](images/Confusion_Matrix_normalized_after_graph_improvement.png)

## Images

### Image 1

![](images/test_image_0.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 44 |
| TP | 354 |
| FN | 76 |
| Precision | 0.8894472361809045 |
| Recall | 0.8232558139534883 |
| Accuracy | 0.7468354430379747 |
| F1 | 0.855072463768116 |
| n_true | 430 |
| n_pred | 398 |
| mean_true_score | 0.6505118436591569 |
| mean_matched_score | 0.790169753597281 |
| panoptic_quality | 0.6756523980034722 |

![](images/test_image_0_classes.png)

![](images/test_image_0_graph.png)

![](images/test_image_0_diff.png)
### Image 2

![](images/test_image_1.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 41 |
| TP | 351 |
| FN | 83 |
| Precision | 0.8954081632653061 |
| Recall | 0.8087557603686636 |
| Accuracy | 0.7389473684210527 |
| F1 | 0.8498789346246973 |
| n_true | 434 |
| n_pred | 392 |
| mean_true_score | 0.6415487544327837 |
| mean_matched_score | 0.7932540154525017 |
| panoptic_quality | 0.6741698775395354 |

![](images/test_image_1_classes.png)

![](images/test_image_1_graph.png)

![](images/test_image_1_diff.png)
### Image 3

![](images/test_image_2.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 71 |
| TP | 488 |
| FN | 87 |
| Precision | 0.8729874776386404 |
| Recall | 0.8486956521739131 |
| Accuracy | 0.7554179566563467 |
| F1 | 0.8606701940035273 |
| n_true | 575 |
| n_pred | 559 |
| mean_true_score | 0.6766823412024456 |
| mean_matched_score | 0.7973203815397669 |
| panoptic_quality | 0.6862298874627977 |

![](images/test_image_2_classes.png)

![](images/test_image_2_graph.png)

![](images/test_image_2_diff.png)
### Image 4

![](images/test_image_3.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 90 |
| TP | 504 |
| FN | 83 |
| Precision | 0.8484848484848485 |
| Recall | 0.858603066439523 |
| Accuracy | 0.7444608567208272 |
| F1 | 0.8535139712108383 |
| n_true | 587 |
| n_pred | 594 |
| mean_true_score | 0.6926210783612383 |
| mean_matched_score | 0.8066836765834263 |
| panoptic_quality | 0.6885157883116797 |

![](images/test_image_3_classes.png)

![](images/test_image_3_graph.png)

![](images/test_image_3_diff.png)
### Image 5

![](images/test_image_4.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 49 |
| TP | 382 |
| FN | 73 |
| Precision | 0.8863109048723898 |
| Recall | 0.8395604395604396 |
| Accuracy | 0.7579365079365079 |
| F1 | 0.8623024830699775 |
| n_true | 455 |
| n_pred | 431 |
| mean_true_score | 0.6639601487379808 |
| mean_matched_score | 0.7908425855386944 |
| panoptic_quality | 0.6819455252274972 |

![](images/test_image_4_classes.png)

![](images/test_image_4_graph.png)

![](images/test_image_4_diff.png)
### Image 6

![](images/test_image_5.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 60 |
| TP | 440 |
| FN | 79 |
| Precision | 0.88 |
| Recall | 0.8477842003853564 |
| Accuracy | 0.7599309153713298 |
| F1 | 0.8635917566241413 |
| n_true | 519 |
| n_pred | 500 |
| mean_true_score | 0.6750925758670521 |
| mean_matched_score | 0.7963023792613636 |
| panoptic_quality | 0.6876801705103042 |

![](images/test_image_5_classes.png)

![](images/test_image_5_graph.png)

![](images/test_image_5_diff.png)
### Image 7

![](images/test_image_6.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 75 |
| TP | 521 |
| FN | 81 |
| Precision | 0.8741610738255033 |
| Recall | 0.8654485049833887 |
| Accuracy | 0.7695716395864106 |
| F1 | 0.8697829716193656 |
| n_true | 602 |
| n_pred | 596 |
| mean_true_score | 0.6968072530043086 |
| mean_matched_score | 0.8051400504963412 |
| panoptic_quality | 0.7002971056904737 |

![](images/test_image_6_classes.png)

![](images/test_image_6_graph.png)

![](images/test_image_6_diff.png)
### Image 8

![](images/test_image_7.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 94 |
| TP | 557 |
| FN | 78 |
| Precision | 0.8556067588325653 |
| Recall | 0.8771653543307086 |
| Accuracy | 0.7640603566529492 |
| F1 | 0.8662519440124417 |
| n_true | 635 |
| n_pred | 651 |
| mean_true_score | 0.712043197511688 |
| mean_matched_score | 0.8117548122440249 |
| panoptic_quality | 0.7031841841678411 |

![](images/test_image_7_classes.png)

![](images/test_image_7_graph.png)

![](images/test_image_7_diff.png)
### Image 9

![](images/test_image_8.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 66 |
| TP | 453 |
| FN | 66 |
| Precision | 0.8728323699421965 |
| Recall | 0.8728323699421965 |
| Accuracy | 0.7743589743589744 |
| F1 | 0.8728323699421965 |
| n_true | 519 |
| n_pred | 519 |
| mean_true_score | 0.7016195837472905 |
| mean_matched_score | 0.803842304558154 |
| panoptic_quality | 0.7016195837472905 |

![](images/test_image_8_classes.png)

![](images/test_image_8_graph.png)

![](images/test_image_8_diff.png)
### Image 10

![](images/test_image_9.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 60 |
| TP | 526 |
| FN | 73 |
| Precision | 0.8976109215017065 |
| Recall | 0.8781302170283807 |
| Accuracy | 0.7981790591805766 |
| F1 | 0.8877637130801688 |
| n_true | 599 |
| n_pred | 586 |
| mean_true_score | 0.7058445798335768 |
| mean_matched_score | 0.8038039987078185 |
| panoptic_quality | 0.7135880224815401 |

![](images/test_image_9_classes.png)

![](images/test_image_9_graph.png)

![](images/test_image_9_diff.png)
### Image 11

![](images/test_image_10.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 66 |
| TP | 555 |
| FN | 70 |
| Precision | 0.893719806763285 |
| Recall | 0.888 |
| Accuracy | 0.8031837916063675 |
| F1 | 0.8908507223113965 |
| n_true | 625 |
| n_pred | 621 |
| mean_true_score | 0.719120166015625 |
| mean_matched_score | 0.8098200067743525 |
| panoptic_quality | 0.7214287379771519 |

![](images/test_image_10_classes.png)

![](images/test_image_10_graph.png)

![](images/test_image_10_diff.png)
### Image 12

![](images/test_image_11.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 83 |
| TP | 585 |
| FN | 71 |
| Precision | 0.875748502994012 |
| Recall | 0.8917682926829268 |
| Accuracy | 0.7916102841677943 |
| F1 | 0.8836858006042296 |
| n_true | 656 |
| n_pred | 668 |
| mean_true_score | 0.7285018083525867 |
| mean_matched_score | 0.8169182671440972 |
| panoptic_quality | 0.7218990729294514 |

![](images/test_image_11_classes.png)

![](images/test_image_11_graph.png)

![](images/test_image_11_diff.png)
### Image 13

![](images/test_image_12.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 65 |
| TP | 452 |
| FN | 69 |
| Precision | 0.874274661508704 |
| Recall | 0.8675623800383877 |
| Accuracy | 0.7713310580204779 |
| F1 | 0.8709055876685935 |
| n_true | 521 |
| n_pred | 517 |
| mean_true_score | 0.6987536644523752 |
| mean_matched_score | 0.8054218123444413 |
| panoptic_quality | 0.7014463568009393 |

![](images/test_image_12_classes.png)

![](images/test_image_12_graph.png)

![](images/test_image_12_diff.png)
### Image 14

![](images/test_image_13.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 61 |
| TP | 527 |
| FN | 80 |
| Precision | 0.8962585034013606 |
| Recall | 0.8682042833607908 |
| Accuracy | 0.7889221556886228 |
| F1 | 0.8820083682008368 |
| n_true | 607 |
| n_pred | 588 |
| mean_true_score | 0.6982983458768791 |
| mean_matched_score | 0.8043018898430088 |
| panoptic_quality | 0.7094009974012814 |

![](images/test_image_13_classes.png)

![](images/test_image_13_graph.png)

![](images/test_image_13_diff.png)
### Image 15

![](images/test_image_14.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 60 |
| TP | 549 |
| FN | 68 |
| Precision | 0.9014778325123153 |
| Recall | 0.8897893030794165 |
| Accuracy | 0.810930576070901 |
| F1 | 0.8955954323001631 |
| n_true | 617 |
| n_pred | 609 |
| mean_true_score | 0.7198640647158631 |
| mean_matched_score | 0.8090275554274818 |
| panoptic_quality | 0.7245613832458198 |

![](images/test_image_14_classes.png)

![](images/test_image_14_graph.png)

![](images/test_image_14_diff.png)
### Image 16

![](images/test_image_15.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 83 |
| TP | 581 |
| FN | 69 |
| Precision | 0.875 |
| Recall | 0.8938461538461538 |
| Accuracy | 0.7926330150068213 |
| F1 | 0.8843226788432268 |
| n_true | 650 |
| n_pred | 664 |
| mean_true_score | 0.7307999830979567 |
| mean_matched_score | 0.8175903425364404 |
| panoptic_quality | 0.7230136819081764 |

![](images/test_image_15_classes.png)

![](images/test_image_15_graph.png)

![](images/test_image_15_diff.png)
### Image 17

![](images/test_image_16.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 96 |
| TP | 542 |
| FN | 73 |
| Precision | 0.8495297805642633 |
| Recall | 0.8813008130081301 |
| Accuracy | 0.7623066104078763 |
| F1 | 0.86512370311253 |
| n_true | 615 |
| n_pred | 638 |
| mean_true_score | 0.7160209841844513 |
| mean_matched_score | 0.8124592348218405 |
| panoptic_quality | 0.7028777418570431 |

![](images/test_image_16_classes.png)

![](images/test_image_16_graph.png)

![](images/test_image_16_diff.png)
### Image 18

![](images/test_image_17.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 103 |
| TP | 543 |
| FN | 69 |
| Precision | 0.8405572755417957 |
| Recall | 0.8872549019607843 |
| Accuracy | 0.7594405594405594 |
| F1 | 0.863275039745628 |
| n_true | 612 |
| n_pred | 646 |
| mean_true_score | 0.7204940895629085 |
| mean_matched_score | 0.8120485871316758 |
| panoptic_quality | 0.7010212763314785 |

![](images/test_image_17_classes.png)

![](images/test_image_17_graph.png)

![](images/test_image_17_diff.png)
### Image 19

![](images/test_image_18.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 91 |
| TP | 563 |
| FN | 62 |
| Precision | 0.8608562691131498 |
| Recall | 0.9008 |
| Accuracy | 0.7863128491620112 |
| F1 | 0.8803752931978108 |
| n_true | 625 |
| n_pred | 654 |
| mean_true_score | 0.73145380859375 |
| mean_matched_score | 0.8120046720623335 |
| panoptic_quality | 0.714868851244869 |

![](images/test_image_18_classes.png)

![](images/test_image_18_graph.png)

![](images/test_image_18_diff.png)
### Image 20

![](images/test_image_19.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 92 |
| TP | 556 |
| FN | 61 |
| Precision | 0.8580246913580247 |
| Recall | 0.9011345218800648 |
| Accuracy | 0.7842031029619182 |
| F1 | 0.8790513833992095 |
| n_true | 617 |
| n_pred | 648 |
| mean_true_score | 0.7320490362578504 |
| mean_matched_score | 0.8123637686530463 |
| panoptic_quality | 0.7141094946578558 |

![](images/test_image_19_classes.png)

![](images/test_image_19_graph.png)

![](images/test_image_19_diff.png)
### Image 21

![](images/test_image_20.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 38 |
| TP | 389 |
| FN | 46 |
| Precision | 0.9110070257611241 |
| Recall | 0.8942528735632184 |
| Accuracy | 0.8224101479915433 |
| F1 | 0.9025522041763341 |
| n_true | 435 |
| n_pred | 427 |
| mean_true_score | 0.7133600344603089 |
| mean_matched_score | 0.7977162339080576 |
| panoptic_quality | 0.7199805452209614 |

![](images/test_image_20_classes.png)

![](images/test_image_20_graph.png)

![](images/test_image_20_diff.png)
### Image 22

![](images/test_image_21.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 30 |
| TP | 359 |
| FN | 42 |
| Precision | 0.922879177377892 |
| Recall | 0.8952618453865336 |
| Accuracy | 0.8329466357308585 |
| F1 | 0.9088607594936708 |
| n_true | 401 |
| n_pred | 389 |
| mean_true_score | 0.7093103270875546 |
| mean_matched_score | 0.7922937079724495 |
| panoptic_quality | 0.7200846611698971 |

![](images/test_image_21_classes.png)

![](images/test_image_21_graph.png)

![](images/test_image_21_diff.png)
### Image 23

![](images/test_image_22.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 25 |
| TP | 353 |
| FN | 44 |
| Precision | 0.9338624338624338 |
| Recall | 0.889168765743073 |
| Accuracy | 0.8364928909952607 |
| F1 | 0.9109677419354839 |
| n_true | 397 |
| n_pred | 378 |
| mean_true_score | 0.7010153667152078 |
| mean_matched_score | 0.7883940526513811 |
| panoptic_quality | 0.7182015498991936 |

![](images/test_image_22_classes.png)

![](images/test_image_22_graph.png)

![](images/test_image_22_diff.png)
### Image 24

![](images/test_image_23.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 44 |
| TP | 388 |
| FN | 50 |
| Precision | 0.8981481481481481 |
| Recall | 0.8858447488584474 |
| Accuracy | 0.8049792531120332 |
| F1 | 0.8919540229885058 |
| n_true | 438 |
| n_pred | 432 |
| mean_true_score | 0.6970701174104594 |
| mean_matched_score | 0.786898740788096 |
| panoptic_quality | 0.7018774975305316 |

![](images/test_image_23_classes.png)

![](images/test_image_23_graph.png)

![](images/test_image_23_diff.png)
### Image 25

![](images/test_image_24.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 44 |
| TP | 366 |
| FN | 55 |
| Precision | 0.8926829268292683 |
| Recall | 0.8693586698337292 |
| Accuracy | 0.7870967741935484 |
| F1 | 0.8808664259927798 |
| n_true | 421 |
| n_pred | 410 |
| mean_true_score | 0.6835079238420427 |
| mean_matched_score | 0.7862208632172131 |
| panoptic_quality | 0.6925555618231047 |

![](images/test_image_24_classes.png)

![](images/test_image_24_graph.png)

![](images/test_image_24_diff.png)
### Image 26

![](images/test_image_25.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 43 |
| TP | 379 |
| FN | 57 |
| Precision | 0.8981042654028436 |
| Recall | 0.8692660550458715 |
| Accuracy | 0.791231732776618 |
| F1 | 0.8834498834498834 |
| n_true | 436 |
| n_pred | 422 |
| mean_true_score | 0.6802926719735521 |
| mean_matched_score | 0.7826058178904188 |
| panoptic_quality | 0.6913930186024912 |

![](images/test_image_25_classes.png)

![](images/test_image_25_graph.png)

![](images/test_image_25_diff.png)
### Image 27

![](images/test_image_26.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 51 |
| TP | 384 |
| FN | 56 |
| Precision | 0.8827586206896552 |
| Recall | 0.8727272727272727 |
| Accuracy | 0.7820773930753564 |
| F1 | 0.8777142857142857 |
| n_true | 440 |
| n_pred | 435 |
| mean_true_score | 0.6891946965997869 |
| mean_matched_score | 0.7897022565205892 |
| panoptic_quality | 0.6931329520089285 |

![](images/test_image_26_classes.png)

![](images/test_image_26_graph.png)

![](images/test_image_26_diff.png)
### Image 28

![](images/test_image_27.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 43 |
| TP | 394 |
| FN | 50 |
| Precision | 0.9016018306636155 |
| Recall | 0.8873873873873874 |
| Accuracy | 0.8090349075975359 |
| F1 | 0.8944381384790011 |
| n_true | 444 |
| n_pred | 437 |
| mean_true_score | 0.6977656596415752 |
| mean_matched_score | 0.7863146012204553 |
| panoptic_quality | 0.7033097681744821 |

![](images/test_image_27_classes.png)

![](images/test_image_27_graph.png)

![](images/test_image_27_diff.png)
### Image 29

![](images/test_image_28.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 45 |
| TP | 386 |
| FN | 43 |
| Precision | 0.8955916473317865 |
| Recall | 0.8997668997668997 |
| Accuracy | 0.8143459915611815 |
| F1 | 0.8976744186046511 |
| n_true | 429 |
| n_pred | 431 |
| mean_true_score | 0.7106071418815559 |
| mean_matched_score | 0.7897680411067034 |
| panoptic_quality | 0.7089545671329942 |

![](images/test_image_28_classes.png)

![](images/test_image_28_graph.png)

![](images/test_image_28_diff.png)
### Image 30

![](images/test_image_29.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 34 |
| TP | 353 |
| FN | 46 |
| Precision | 0.9121447028423773 |
| Recall | 0.8847117794486216 |
| Accuracy | 0.815242494226328 |
| F1 | 0.8982188295165394 |
| n_true | 399 |
| n_pred | 387 |
| mean_true_score | 0.7014908181097275 |
| mean_matched_score | 0.7929032193364908 |
| panoptic_quality | 0.7122006015923187 |

![](images/test_image_29_classes.png)

![](images/test_image_29_graph.png)

![](images/test_image_29_diff.png)
