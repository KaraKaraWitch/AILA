import enum
import json
from typing_extensions import Annotated
from sam_lstm import SalMap
import sys, pathlib
import typer


app_help = (
    
)

app = typer.Typer(help=app_help, add_completion=False)

k_count_help = "Set's the number of 'hotspots' to detect."

k_reverse_help = (
    "Enables 'Greedy' selection of K.\n"
    "It tries to get as much points as possible before trying even lesser.\n"
    "This has an effect by flipping max gaps."
).replace("\n", " ")

visual_help = (
    "Saves an additional image showing centroids (red dots and blue boxes [Crops])"
).replace("\n", "")

max_gap_help = (
    "Determines how wide in terms of spread the saliency will need before\n"
    "it is picked up as the center.\n"
    "The lower value picks up more spots, higher values means tighter spots.\n"
    "Note that reverse_k will reverse this value! (0.2 -> 0.8)"
).replace("\n", " ")

peak_threshold_help = (
    "Determines how high the peak for a 'hotspot'/'centroid' needs to be before it gets picked up.\n"
    "The lower value picks up more spots, higher values means tighter spots.\n"
).replace("\n", " ")

aspect_help = (
    "Determines how wide to tall the crop box should be.\n"
    "Same as the aspect ratio monitor screen.\n"
    "(1.0 is Square), (0.0 is 'auto')\n"
    "Auto means that it will take the aspect ratio from the image itself."
).replace("\n", " ")


@app.command()
def kara_split(
    image: pathlib.Path,
    k_count: int = 4,
    reverse_k: bool = True,
    visualize: bool = True,
    max_gap: float = 0.98,
    peak_thr: float = 0.3,
    aspect_ratio: float = 0.3,
):

    if aspect_ratio == 0.0:
        aspect_ratio = "auto"  # noqa
    elif aspect_ratio < 0.0:
        raise Exception("Aspect ratio cannot be negative.")

    print("Loading weights...")
    salmap = SalMap()
    salmap.compile()
    salmap.load_weights()
    print("Weights loaded")

    # Crop args optmized for selection of multiple parts.
    crop_args = {
        "max_gap": max_gap,  # Determines how wide in terms of spread the saliency will need before it is picked up as the center. (Defaults to 0.2.). The lower value picks up more spots, higher values means tighter spots.
        "peak_thr": peak_thr,  # Determines how high the peak for a "hotspot"/"centroid" needs to be before it gets picked up. (Defaults to 0.5.). Lower means that it will also tend to bias lower thereshold saliency.
        "dsc_asp": aspect_ratio,  # Determines how wide to tall the crop box should be. It's like the aspect ratio similar to a monitor screen. (1.44 which is around 4448x3096px). (1.0 is Square). special value "auto" will automatically calculate the aspect ratio for you in respect to the image.
        "reverse_k": reverse_k,  # Reverses the no. of centroid to retrieve, this is useful if you need more images/points but max_gap wasn't giving it to you. It does slow the the process... but you get more points so yay? (This essentially reverses max_gap)
        "max_k": k_count,  # No of centroids to retrieve per image. Default is 4.
    }

    salmap.shinon_predicts(
        [pathlib.Path(sys.argv[1])], visualize=visualize, crop_args=crop_args
    )


doc = f"""

Aira Is Lookin Around (A toolkit to detect salient images for Stable Diffusion).
At it's core, AILA uses SAM-LSTM (Predicting Human Eye Fixations via an LSTM-based Saliency Attentive Model) with some additional features that would be of use for others.
These scripts were created due to myself finding the need to crop large images (with only 1 face) to faces or other attentive parts.


--k_count [float]: 

{k_count_help}

--reverse_k [bool]:

{k_reverse_help}

--visualize [bool]:

{visual_help}

--max_gap [float]:

{max_gap_help}

--peak_thr [float]:

{peak_threshold_help}

--aspect_ratio [float]:

{aspect_help}
"""
kara_split.__doc__ = doc


if __name__ == "__main__":
    app()
