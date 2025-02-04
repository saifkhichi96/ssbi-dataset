import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_coco(coco_path: str) -> Dict[str, Any]:
    """
    Load a COCO dataset annotation file.

    Args:
        coco_path (str): Path to the COCO JSON file.

    Returns:
        Dict[str, Any]: Parsed COCO annotation data.
    """
    coco_file = Path(coco_path)
    if not coco_file.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {coco_path}")

    with open(coco_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_category_id(coco: Dict[str, Any], category_name: str) -> int:
    """
    Retrieve the category ID corresponding to a given category name.

    Args:
        coco (Dict[str, Any]): COCO dataset dictionary.
        category_name (str): Name of the category.

    Returns:
        int: The category ID, or -1 if not found.
    """
    for category in coco.get("categories", []):
        if category["name"] == category_name:
            return category["id"]
    return -1


def count_instances(coco: Dict[str, Any], category_name: str) -> int:
    """
    Count the number of annotations for a given category.

    Args:
        coco (Dict[str, Any]): COCO dataset dictionary.
        category_name (str): Category name.

    Returns:
        int: The number of instances in the dataset for the specified category.
    """
    category_id = get_category_id(coco, category_name)
    if category_id == -1:
        return 0

    return sum(1 for annotation in coco.get("annotations", []) if annotation["category_id"] == category_id)


def count_instances_by_area(coco: Dict[str, Any], category_name: str, area_range: Tuple[float, float]) -> int:
    """
    Count the number of annotations for a category within a specific area range.

    Args:
        coco (Dict[str, Any]): COCO dataset dictionary.
        category_name (str): Category name.
        area_range (Tuple[float, float]): Range of area (min, max).

    Returns:
        int: The number of instances within the given area range.
    """
    category_id = get_category_id(coco, category_name)
    if category_id == -1:
        return 0

    return sum(
        1
        for annotation in coco.get("annotations", [])
        if annotation["category_id"] == category_id and area_range[0] <= annotation["bbox"][2] * annotation["bbox"][3] < area_range[1]
    )


def print_statistics(train: Dict[str, Any], val: Dict[str, Any]) -> None:
    """
    Print dataset statistics including category-wise instance count and distribution by area.

    Args:
        train (Dict[str, Any]): COCO annotations for the training set.
        val (Dict[str, Any]): COCO annotations for the validation set.
    """
    # Count genuine and forged signatures
    num_train_genuine = count_instances(train, "signature_g")
    num_train_forged = count_instances(train, "signature_f")
    num_val_genuine = count_instances(val, "signature_g")
    num_val_forged = count_instances(val, "signature_f")

    num_total_genuine = num_train_genuine + num_val_genuine
    num_total_forged = num_train_forged + num_val_forged

    # Table 1: Data splits
    logging.info("\nTable 1: Detailed Data Splits for Genuine and Forged Signatures")
    logging.info("-" * 50)
    logging.info("\t\tTrain\tVal\tTotal")
    logging.info("-" * 50)
    logging.info(f"Genuine\t\t{num_train_genuine}\t{num_val_genuine}\t{num_total_genuine}")
    logging.info(f"Forged\t\t{num_train_forged}\t{num_val_forged}\t{num_total_forged}")
    logging.info("-" * 50)
    logging.info(f"Total\t\t{num_train_genuine + num_train_forged}\t"
                 f"{num_val_genuine + num_val_forged}\t"
                 f"{num_total_genuine + num_total_forged}")
    logging.info("-" * 50)

    # Define area ranges
    small_area = 32 * 32
    medium_area = 96 * 96
    area_ranges = {
        "small": (0, small_area),
        "medium": (small_area, medium_area),
        "large": (medium_area, float("inf")),
    }

    # Table 2: Area distribution
    logging.info("\nTable 2: Distribution of Small, Medium, and Large Annotations")
    logging.info("-" * 70)
    logging.info("Category\t\tTrain\t\t\tVal")
    logging.info("\t\tSmall\tMedium\tLarge\tSmall\tMedium\tLarge")
    logging.info("-" * 70)

    for category in train.get("categories", []):
        category_name = category["name"]
        if category_name == "signature":
            continue

        train_counts = [count_instances_by_area(train, category_name, area_ranges[size]) for size in area_ranges]
        val_counts = [count_instances_by_area(val, category_name, area_ranges[size]) for size in area_ranges]

        logging.info(f"{category_name:<15}\t{train_counts[0]:>5}\t{train_counts[1]:>6}\t{train_counts[2]:>5}"
                     f"\t{val_counts[0]:>5}\t{val_counts[1]:>6}\t{val_counts[2]:>5}")
    logging.info("-" * 70)


if __name__ == "__main__":
    # Load COCO dataset
    train_coco_path = "data/ssbi/annotations/instances_train.json"
    val_coco_path = "data/ssbi/annotations/instances_val.json"

    try:
        coco_train = load_coco(train_coco_path)
        coco_val = load_coco(val_coco_path)
        print_statistics(coco_train, coco_val)
    except FileNotFoundError as e:
        logging.error(e)
