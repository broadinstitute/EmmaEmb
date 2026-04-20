import pandas as pd

from emmaemb.core import Emma
from emmaemb.visualization import (
    plot_emb_dis_scatter, 
    plot_knn_alignment_across_embedding_spaces,
)


# parameter for this script
figures_to_be_plotted = [
    #'Fig_A1',
    'Fig_A2',
    'Fig_A3',
    #'Fig_B1',
    #"Fig_B2",
    #"Fig_B3",
]

output_dir = "figures/"
distance_metric = "cosine"
k_neighbors = 10

fp_metadata = "examples/Pla2g2/Pla2g2_features.csv"
embedding_dir = "embeddings/"
models = {
    "ProtT5": "Rostlab/prot_t5_xl_uniref50/layer_None/chopped_1022_overlap_300",
    "ESMC": "esmc-300m-2024-12/layer_None/chopped_1022_overlap_300",
}

metadata = pd.read_csv(fp_metadata)
ema = Emma(feature_data=metadata)

for model_alias, model_name in models.items():
    ema.add_emb_space(
        embeddings_source=embedding_dir + model_name,
        emb_space_name=model_alias,
    )
    
ema.mean_center_emb_spaces()

if "Fig_A1" in figures_to_be_plotted:

    fig_A1 = ema.plot_emb_dis_scatter(
        emb_space_name_1="ESMC",
        emb_space_name_2="ProtT5",
        distance_metric=distance_metric,
    )

    fig_A1.update_layout(
        title=None,
        # title="A. All pairwise distances between PLA2G2 embeddings<br>All embedding paris.",
        title_font=dict(size=26),
        xaxis_title="Cosine distances ESMC",
        xaxis_title_font=dict(size=26),
        yaxis_title="Cosine distances ProtT5",
        yaxis_title_font=dict(size=26),
        xaxis_tickfont=dict(size=26),
        yaxis_tickfont=dict(size=26),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig_A1.update_traces(marker=dict(size=8, color="grey"))

    fig_A1.write_image(
        output_dir + "fig_4_A1.pdf", format="pdf", width=600, height=600
    )

if "Fig_A2" in figures_to_be_plotted:

    fig_A2 = ema.plot_emb_dis_scatter(
        emb_space_name_1="ESMC",
        emb_space_name_2="ProtT5",
        distance_metric=distance_metric,
        color_group="species",
        color_value_1="birds",
        color_value_2="birds",
    )

    fig_A2.update_layout(
        title=None,
        # title="A. All pairwise distances between PLA2G2 embeddings<br>All embedding paris.",
        title_font=dict(size=26),
        xaxis_title="Cosine distances ESMC",
        xaxis_title_font=dict(size=26),
        yaxis_title="Cosine distances ProtT5",
        yaxis_title_font=dict(size=26),
        xaxis_tickfont=dict(size=26),
        yaxis_tickfont=dict(size=26),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig_A2.update_traces(marker=dict(size=8))
    fig_A2.data[1].marker.color = "darkred"

    fig_A2.write_image(
        output_dir + "fig_4_A2.pdf", format="pdf", width=600, height=600
    )

if "Fig_A3" in figures_to_be_plotted:

    fig_A3 = ema.plot_emb_dis_scatter(
        emb_space_name_1="ESMC",
        emb_space_name_2="ProtT5",
        distance_metric=distance_metric,
        color_group="species",
        color_value_1="crocodile",
        color_value_2="crocodile",
    )

    fig_A3.update_layout(
        title=None,
        # title="A. All pairwise distances between PLA2G2 embeddings<br>All embedding paris.",
        title_font=dict(size=26),
        xaxis_title="Cosine distances ESMC",
        xaxis_title_font=dict(size=26),
        yaxis_title="Cosine distances ProtT5",
        yaxis_title_font=dict(size=26),
        xaxis_tickfont=dict(size=26),
        yaxis_tickfont=dict(size=26),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig_A3.update_traces(marker=dict(size=8))

    fig_A3.data[1].marker.color = "orange"

    fig_A3.write_image(
        output_dir + "fig_4_A3.pdf", format="pdf", width=600, height=600
    )


if "Fig_B1" in figures_to_be_plotted:

    fig_B1 = analyze_low_similarity_distribution(
        ema,
        "ESMC",
        "ProtT5",
        "species",
        similarity_threshold=0.3,
        k_neighbors=10,
        distance_metric="cosine",
    )

    tick_vals = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    fig_B1.update_layout(
        legend=dict(
            font=dict(size=18),
            title="Species",
        ),
        title=None,
        # title="A. All pairwise distances between PLA2G2 embeddings<br>All embedding paris.",
        title_font=dict(size=26),
        xaxis_title="% in dataset",
        xaxis_title_font=dict(size=26),
        yaxis_title="% in low similarity subset",
        yaxis_title_font=dict(size=26),
        xaxis_tickfont=dict(size=26),
        yaxis_tickfont=dict(size=26),
        xaxis=dict(
            range=[-0.05, 0.7],
            tickvals=tick_vals,
            tickformat=".0%",
            dtick=0.2,
            tickwidth=6,
            tickcolor="black",
        ),
        yaxis=dict(range=[-0.05, 0.7], tickformat=".0%", dtick=0.2),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig_B1.update_traces(marker=dict(size=12))

    fig_B1.write_image(
        output_dir + "fig_4_B1.pdf", format="pdf", width=600, height=400
    )

if "Fig_B2" in figures_to_be_plotted:

    fig_B2 = analyze_low_similarity_distribution(
        ema,
        "ESMC",
        "ProtT5",
        "enzyme_class",
        similarity_threshold=0.3,
        k_neighbors=10,
        distance_metric="cosine",
    )

    tick_vals = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    fig_B2.update_layout(
        legend=dict(
            font=dict(size=18),
            title="Enzyme class",
        ),
        title=None,
        # title="A. All pairwise distances between PLA2G2 embeddings<br>All embedding paris.",
        title_font=dict(size=26),
        xaxis_title="% in dataset",
        xaxis_title_font=dict(size=26),
        yaxis_title="% in low similarity subset",
        yaxis_title_font=dict(size=26),
        xaxis_tickfont=dict(size=26),
        yaxis_tickfont=dict(size=26),
        xaxis=dict(
            range=[-0.05, 0.4],
            tickvals=tick_vals,
            tickformat=".0%",
            dtick=0.2,
            tickwidth=6,
            tickcolor="black",
        ),
        yaxis=dict(range=[-0.05, 0.4], tickformat=".0%", dtick=0.2),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig_B2.update_traces(marker=dict(size=12))

    fig_B2.write_image(
        output_dir + "fig_4_B2.pdf", format="pdf", width=600, height=400
    )

if "Fig_B3" in figures_to_be_plotted:

    fig_B3 = analyze_low_similarity_distribution(
        ema,
        "ESMC",
        "ProtT5",
        "length_bin",
        similarity_threshold=0.3,
        k_neighbors=10,
        distance_metric="cosine",
    )

    tick_vals = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    fig_B3.update_layout(
        legend=dict(
            font=dict(size=18),
            title="Sequence length",
        ),
        title=None,
        # title="A. All pairwise distances between PLA2G2 embeddings<br>All embedding paris.",
        title_font=dict(size=26),
        xaxis_title="% in dataset",
        xaxis_title_font=dict(size=26),
        yaxis_title="% in low similarity subset",
        yaxis_title_font=dict(size=26),
        xaxis_tickfont=dict(size=26),
        yaxis_tickfont=dict(size=26),
        xaxis=dict(
            range=[-0.05, 0.9],
            tickvals=tick_vals,
            tickformat=".0%",
            dtick=0.2,
            tickwidth=6,
            tickcolor="black",
        ),
        yaxis=dict(range=[-0.05, 0.9], tickformat=".0%", dtick=0.2),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig_B3.update_traces(marker=dict(size=12))

    fig_B3.write_image(
        output_dir + "fig_4_B3.pdf", format="pdf", width=600, height=400
    )

    print()
