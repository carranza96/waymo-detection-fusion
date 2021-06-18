import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def show_values_on_bars(axs):
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            if p.get_height() < 1:
                value = '{:.3f}'.format(p.get_height())
            else:
                value = '{:.0f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center", fontsize=7 )

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)



def barplot_diff_results(file, legend_labels=None, title=None):
    df = pd.read_csv("stats_res/"+file+".csv")
    df["TPdets%"] = df["TP"]/df["dets"]
    df["FPdets%"] = df["FP"]/df["dets"]
    df = df.reindex(columns=['model_name', 'class', 'gts', 'dets', 'recall', 'ap', 'TP', "TPdets%", "FPdets%", 'FP', 'FPRed', 'FN'])
    print()

    fig = plt.figure(figsize=(18, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    sns.set_theme(style="whitegrid")
    for i, col in enumerate(df.columns[3:]):
        if i==2 or i==5:
            ax = fig.add_subplot(3, 3, i+1, sharey=ax)
        else:
            ax = fig.add_subplot(3, 3, i+1)

        g = sns.barplot(data=df[df["class"]=="All"], x="class", y=col, hue="model_name", ax=ax, palette="Blues_d")
        ax.legend([], [], frameon=False)
        show_values_on_bars(ax)

    handles, labels = ax.get_legend_handles_labels()
    if legend_labels:
        labels = legend_labels
    fig.legend(handles, labels, loc='lower center', ncol=3)
    if title:
        fig.suptitle(title)
    # plt.show()
    plt.tight_layout()
    fig.savefig("stats_res/"+file+".png", dpi=500)




def plots_dets_gts_stats(model="faster_rcnn_r50_fpn_fp16_4x1_3e_1280x1920"):
    df_dets = pd.read_csv("stats_res/{}/det_stats.csv".format(model))
    df_gts = pd.read_csv("stats_res/{}/det_stats.csv".format(model))

    # df_dets = df_dets
    # fig = plt.figure(figsize=(18, 10))
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)
    # # sns.set_theme(style="whitegrid")
    #
    # ax = fig.add_subplot(3, 2, 1)
    # sns.histplot(data=df_dets[df_dets["fp"] == 1], x="score", hue="class", bins=20, ax=ax, palette="Blues_d")
    # ax.set_title("False positives score distribution")
    #
    # ax = fig.add_subplot(3, 2, 2)
    # sns.histplot(data=df_dets[df_dets["tp"] == 1], x="score", hue="class", bins=20, ax=ax, palette="Blues_d")
    # ax.set_title("True positives score distribution")
    #
    #
    # ax = fig.add_subplot(3, 2, 3)
    # sns.scatterplot(data=df_dets[df_dets["fp"] == 1], x="score", y="height", hue="class", ax=ax, palette="Blues_d", s=2)
    # ax.set_title("False positives score/height distribution")
    #
    # ax = fig.add_subplot(3, 2, 4)
    # # sns.scatterplot(data=df_dets[(df_dets["tp"] == 1) & (df_dets["frontal"] == 1)], x="score", y="height", hue="class", ax=ax, palette="Blues_d", s=2)
    # sns.scatterplot(data=df_dets[(df_dets["tp"] == 1)], x="score", y="height", hue="class", ax=ax, palette="Blues_d", s=2)
    # ax.set_title("True positives score/height distribution")
    #
    # ax = fig.add_subplot(3, 2, 5)
    # sns.scatterplot(data=df_dets[(df_dets["tp"] == 1)], x="score", y="iou_matched_gt", hue="class", ax=ax, palette="Blues_d", s=1)
    # ax.set_title("True positives score/IoU distribution")
    #
    # ax = fig.add_subplot(3, 2, 6)
    # sns.scatterplot(data=df_dets[(df_dets["fp"] == 1) & (df_dets["iou_matched_gt"] != -1)], x="score", y="iou_matched_gt", hue="class", ax=ax, palette="Blues_d", s=1)
    # ax.set_title("False positives score/IoU distribution")
    #
    #
    # fig.savefig("stats_res/{}/scores_dist.png".format(model), dpi=300)

    fig = plt.figure(figsize=(12, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    ax = fig.add_subplot(1, 1, 1)
    sns.histplot(data=df_dets[(df_dets["fp"] == 1) & (df_dets["fp_cats"]) & (df_dets["class"] == 0)], x="score",
                     hue="fp_cats", ax=ax, bins=10)
    ax.set_title("False positives")

    fig.savefig("stats_res/{}/fps.png".format(model), dpi=300)


    fig = plt.figure(figsize=(12, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    ax = fig.add_subplot(1, 1, 1)
    sns.histplot(data=df_dets[(df_dets["tp"] == 1) & (df_dets["class"] == 0)], x="score", ax=ax, bins=10)
    ax.set_title("True positives")

    fig.savefig("stats_res/{}/tps.png".format(model), dpi=300)




# plots_dets_gts_stats(model="ensemble/wbf_faster50_retina50_3e")
plots_dets_gts_stats(model="faster_rcnn_r50_fpn_fp16_4x1_3e_1280x1920")

# Different resolution
# barplot_diff_results("res_diff_models",
#                      ["FRCNN R50 640x960", "FRCNN R50 1280x1920"],
#                      "Difference between predictions with different image resolution")
#
# # Different models
# barplot_diff_results("res_diff_models",
#                      ["FRCNN R50", "CASCADE R50", "CASCADE RES2NET"],
#                      "Difference between different models (3 epochs)")
#
# Plot difference between number of epochs
# barplot_diff_results("res_diff_epochs",
#                      ["FRCNN R50 3e", "FRCNN R50 12e", "FRCNN R50 24e"],
#                      "Difference between same model with different number of epochs (test)")

# # Plot difference between score thresholds
# barplot_diff_results("res_diff_score_th",
#                      ["FRCNN R50", "FRCNN R50 0.1", "FRCNN R50 0.3", "FRCNN R50 0.5"],
#                      "Difference between same model with different score threshold on test predictions (3 epochs)")

# barplot_diff_results("ensemble_2m",
#                      ["FRCNN R50", "CASCADE R50", "WBF Ensemble", "NMS Ensemble"],
#                      "Ensemble results (3 epochs)")
#
# barplot_diff_results("ensemble_3m",
#                      ["FRCNN R50", "CASCADE R50", "CASCADE R2NET", "WBF Ensemble", "NMS Ensemble"],
#                      "Ensemble results (3 epochs)")


# def model_stats_plot(model)