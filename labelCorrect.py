import numpy as np
def CorrectLabel_threshold(original_labels, model_predictions, conf_threshold=0.8):

    new_label_dict = {}
    # print(model_predictions.shape)
    for pid in range(model_predictions.shape[0]):
        pred_prob = model_predictions[pid]      #预测原始数据
        pred = np.argmax(pred_prob, axis=0)  #预测0/1二值结果
        label = original_labels[pid]    ##需要校正的label


        new_label_dict[pid] = label
        
        #1：表示前景
        confident = (pred_prob[1] > conf_threshold)
        #前景大于前景的阈值或者背景大于背景的阈值，表明都是自信的，可以替换的

        # before update: only class that need correction will be replaced
        belong_to_correction_class = label==0
        #没有假阳的情况下，只用correct背景

        # after update: only pixels that will be flipped to the allowed classes will be updated
        after_belong = pred==1
        

        # combine all three masks together
        replace_flag = confident & belong_to_correction_class & after_belong

       

        # replace with the new label
        next_label = np.where(replace_flag, pred, label).astype("int32")

       

        new_label_dict[pid] = next_label

    return new_label_dict