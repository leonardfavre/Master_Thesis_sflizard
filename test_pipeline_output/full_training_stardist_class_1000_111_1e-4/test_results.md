# Test results

## Segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 27573 |
| TP | 111292 |
| FN | 22855 |
| Precision | 0.8014402477226082 |
| Recall | 0.8296272000119272 |
| Accuracy | 0.6881770962156815 |
| F1 | 0.8152901703954405 |
| n_true | 134147 |
| n_pred | 138865 |
| mean_true_score | 0.6459432018691217 |
| mean_matched_score | 0.7785945324114677 |
| panoptic_quality | 0.6347804689987039 |
## Segmentation Metrics per class

| Metric | Neutrophil | Epithelial | Lymphocyte | Plasma | Eosinophil | Connective tissue |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| FP |431.0000 |  93.0000 |  363.0000 |  358.0000 |  180.0000 |  223.0000 | 
| TP |0.0000 |  333.0000 |  79.0000 |  3.0000 |  27.0000 |  228.0000 | 
| FN |200.0000 |  59.0000 |  362.0000 |  335.0000 |  175.0000 |  221.0000 | 
| Precision |0.0000 |  0.7817 |  0.1787 |  0.0083 |  0.1304 |  0.5055 | 
| Recall |0.0000 |  0.8495 |  0.1791 |  0.0089 |  0.1337 |  0.5078 | 
| Accuracy |0.0000 |  0.6866 |  0.0983 |  0.0043 |  0.0707 |  0.3393 | 
| F1 |0.0000 |  0.8142 |  0.1789 |  0.0086 |  0.1320 |  0.5067 | 
| n_true |200.0000 |  392.0000 |  441.0000 |  338.0000 |  202.0000 |  449.0000 | 
| n_pred |431.0000 |  426.0000 |  442.0000 |  361.0000 |  207.0000 |  451.0000 | 
| mean_true_score |0.0000 |  0.5466 |  0.1053 |  0.0062 |  0.0900 |  0.2936 | 
| mean_matched_score |0.0000 |  0.6434 |  0.5878 |  0.6944 |  0.6732 |  0.5781 | 
| panoptic_quality |0.0000 |  0.5239 |  0.1052 |  0.0060 |  0.0889 |  0.2929 | 
## Segmentation Metrics per class after graph improvement

| Metric | Neutrophil | Epithelial | Lymphocyte | Plasma | Eosinophil | Connective tissue |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| FP |95.0000 |  7.0000 |  6.0000 |  425.0000 |  446.0000 |  374.0000 | 
| TP |0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  78.0000 | 
| FN |200.0000 |  392.0000 |  441.0000 |  338.0000 |  202.0000 |  371.0000 | 
| Precision |0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.1726 | 
| Recall |0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.1737 | 
| Accuracy |0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0948 | 
| F1 |0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.1731 | 
| n_true |200.0000 |  392.0000 |  441.0000 |  338.0000 |  202.0000 |  449.0000 | 
| n_pred |95.0000 |  7.0000 |  6.0000 |  425.0000 |  446.0000 |  452.0000 | 
| mean_true_score |0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0978 | 
| mean_matched_score |0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.5631 | 
| panoptic_quality |0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0975 | 
## Classification metrics

| Metric | Value |
| :--- | :---: |
| accuracy micro | 0.9434759020805359 |
| f1 micro | 0.9434759020805359 |
| accuracy macro | 0.6209056973457336 |
| f1 macro | 0.5938237905502319 |

## Classification metrics after graph improvement

| Metric | Value |
| :--- | :---: |
| accuracy micro | 0.8792977333068848 |
| f1 micro | 0.8792977333068848 |
| accuracy macro | 0.25845032930374146 |
| f1 macro | 0.20572973787784576 |

## Images

### Image 1

![](images/test_image_0.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 48 |
| TP | 459 |
| FN | 51 |
| Precision | 0.9053254437869822 |
| Recall | 0.9 |
| Accuracy | 0.8225806451612904 |
| F1 | 0.9026548672566371 |
| n_true | 510 |
| n_pred | 507 |
| mean_true_score | 0.7232805439070159 |
| mean_matched_score | 0.8036450487855733 |
| panoptic_quality | 0.7254141148329953 |

![](images/test_image_0_classes.png)

![](images/test_image_0_graph.png)

![](images/test_image_0_diff.png)
### Image 2

![](images/test_image_1.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 37 |
| TP | 407 |
| FN | 42 |
| Precision | 0.9166666666666666 |
| Recall | 0.9064587973273942 |
| Accuracy | 0.8374485596707819 |
| F1 | 0.9115341545352743 |
| n_true | 449 |
| n_pred | 444 |
| mean_true_score | 0.7344863993553382 |
| mean_matched_score | 0.8102810646450783 |
| panoptic_quality | 0.7385988651971934 |

![](images/test_image_1_classes.png)

![](images/test_image_1_graph.png)

![](images/test_image_1_diff.png)
### Image 3

![](images/test_image_2.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 51 |
| TP | 484 |
| FN | 32 |
| Precision | 0.9046728971962616 |
| Recall | 0.937984496124031 |
| Accuracy | 0.8536155202821869 |
| F1 | 0.9210275927687916 |
| n_true | 516 |
| n_pred | 535 |
| mean_true_score | 0.7640175634576368 |
| mean_matched_score | 0.8145311213721914 |
| panoptic_quality | 0.7502056379526939 |

![](images/test_image_2_classes.png)

![](images/test_image_2_graph.png)

![](images/test_image_2_diff.png)
### Image 4

![](images/test_image_3.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 41 |
| TP | 487 |
| FN | 30 |
| Precision | 0.9223484848484849 |
| Recall | 0.941972920696325 |
| Accuracy | 0.8727598566308243 |
| F1 | 0.9320574162679426 |
| n_true | 517 |
| n_pred | 528 |
| mean_true_score | 0.7909629718477998 |
| mean_matched_score | 0.8396875902367813 |
| panoptic_quality | 0.7826370458283493 |

![](images/test_image_3_classes.png)

![](images/test_image_3_graph.png)

![](images/test_image_3_diff.png)
### Image 5

![](images/test_image_4.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 63 |
| TP | 464 |
| FN | 52 |
| Precision | 0.8804554079696395 |
| Recall | 0.8992248062015504 |
| Accuracy | 0.8013816925734024 |
| F1 | 0.8897411313518696 |
| n_true | 516 |
| n_pred | 527 |
| mean_true_score | 0.7359394398770591 |
| mean_matched_score | 0.8184154115874192 |
| panoptic_quality | 0.7281778542215963 |

![](images/test_image_4_classes.png)

![](images/test_image_4_graph.png)

![](images/test_image_4_diff.png)
### Image 6

![](images/test_image_5.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 53 |
| TP | 456 |
| FN | 34 |
| Precision | 0.8958742632612967 |
| Recall | 0.9306122448979591 |
| Accuracy | 0.8397790055248618 |
| F1 | 0.9129129129129129 |
| n_true | 490 |
| n_pred | 509 |
| mean_true_score | 0.7512548329878826 |
| mean_matched_score | 0.8072694477282072 |
| panoptic_quality | 0.7369667030311562 |

![](images/test_image_5_classes.png)

![](images/test_image_5_graph.png)

![](images/test_image_5_diff.png)
### Image 7

![](images/test_image_6.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 57 |
| TP | 495 |
| FN | 31 |
| Precision | 0.8967391304347826 |
| Recall | 0.94106463878327 |
| Accuracy | 0.8490566037735849 |
| F1 | 0.9183673469387755 |
| n_true | 526 |
| n_pred | 552 |
| mean_true_score | 0.7700855559722553 |
| mean_matched_score | 0.8183131362452651 |
| panoptic_quality | 0.751512063898713 |

![](images/test_image_6_classes.png)

![](images/test_image_6_graph.png)

![](images/test_image_6_diff.png)
### Image 8

![](images/test_image_7.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 37 |
| TP | 468 |
| FN | 33 |
| Precision | 0.9267326732673268 |
| Recall | 0.9341317365269461 |
| Accuracy | 0.8698884758364313 |
| F1 | 0.9304174950298211 |
| n_true | 501 |
| n_pred | 505 |
| mean_true_score | 0.7846550551241267 |
| mean_matched_score | 0.8399832961905715 |
| panoptic_quality | 0.7815351543085238 |

![](images/test_image_7_classes.png)

![](images/test_image_7_graph.png)

![](images/test_image_7_diff.png)
### Image 9

![](images/test_image_8.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 64 |
| TP | 475 |
| FN | 50 |
| Precision | 0.8812615955473099 |
| Recall | 0.9047619047619048 |
| Accuracy | 0.8064516129032258 |
| F1 | 0.8928571428571429 |
| n_true | 525 |
| n_pred | 539 |
| mean_true_score | 0.7359373256138393 |
| mean_matched_score | 0.8134044125205592 |
| panoptic_quality | 0.7262539397504993 |

![](images/test_image_8_classes.png)

![](images/test_image_8_graph.png)

![](images/test_image_8_diff.png)
### Image 10

![](images/test_image_9.png)
## Personal segmentation Metrics

| Metric | Value |
| :--- | :---: |
| FP | 53 |
| TP | 498 |
| FN | 39 |
| Precision | 0.9038112522686026 |
| Recall | 0.9273743016759777 |
| Accuracy | 0.8440677966101695 |
| F1 | 0.9154411764705882 |
| n_true | 537 |
| n_pred | 551 |
| mean_true_score | 0.7527347614423301 |
| mean_matched_score | 0.811683869266127 |
| panoptic_quality | 0.7430488362031824 |

![](images/test_image_9_classes.png)

![](images/test_image_9_graph.png)

![](images/test_image_9_diff.png)
