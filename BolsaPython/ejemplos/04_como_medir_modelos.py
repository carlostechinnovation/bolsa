############### MEDIDA DE LA CALIDAD DE LOS MODELOS

# Curva de PRECISION-COBERTURA:
# precision_recall_curve(y_true, probas_pred)
disp = plot_precision_recall_curve(clf_l1_LR, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))

# Curva ROC
roc_curve(y_true, y_score[, pos_label, …])



