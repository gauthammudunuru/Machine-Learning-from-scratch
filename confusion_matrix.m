function [TP,TN,FP,FN, Accuracy,Precision,Recall,F1_Score,Specificity] = confusion_matrix(a,b)

    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0;
    for i=1:length(a)
        
        if(a(i) == 1)
            if(b(i) == 1)
                TP = TP + 1;
            else
                FN = FN + 1;
            end
        else
            if(b(i) == 0)
                TN = TN + 1;
            else
                FP = FP + 1;
            end
        end
        
    end

    Accuracy = (TP + TN) / (TP + FP + TN + FN);
    Precision = TP / (TP + FP);
    Recall = TP / (TP + FN); % Sensitivity
    F1_Score =  2 * (Recall*Precision) / (Recall + Precision);
    Specificity = TN / (TN + FP);

end
