"""Clinical labels and user-facing names for the OCT task."""

CLASS_NAMES = ("CNV", "DME", "DRUSEN", "NORMAL")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
DISPLAY_NAMES = {
    "CNV": "Neovascularización coroidea",
    "DME": "Edema macular diabético",
    "DRUSEN": "Drusas",
    "NORMAL": "Normal",
}
RESEARCH_DISCLAIMER = (
    "Resultado generado por un modelo experimental de investigación. "
    "No sustituye la evaluación de un oftalmólogo."
)
