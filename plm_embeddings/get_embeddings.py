import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Union, Tuple

import numpy as np
import requests
from tqdm import tqdm

from plm_embeddings.embedding_handler_selector import select_embedding_handler
from plm_embeddings.embedding_model_metadata_handler import (
    EmbeddingModelMetadataHandler,
)
from plm_embeddings.utils import (
    read_fasta_names, read_fasta, write_fasta, setup_logger
    )

sys.path.append(str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Get embeddings for protein sequences")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to a FASTA file containing protein sequences or \
            a list of protein names",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Name of the embedding model to be used",
        default="esm3-sm-open-v1",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Output directory for storing embeddings",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        help="Unique identifier for the current run. If not provided, \
            a new run ID will be generated using the current timestamp",
        default=None,
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Flag to disable GPU usage",
        default=False,
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Flag to enable development mode (shortening input data)",
        default=False,
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Layer index to extract embeddings from",
    )
    parser.add_argument(
        "--per_protein",
        action="store_true",
        help="Aggregate embeddings per protein",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="npy",
        help="File format for the embeddings",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1022,
        help="Maximum sequence length. Sequences longer than this will \
            be chopped.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=300,
        help="Overlap size between chunks",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        default=True,
        help="Flag to chop sequences from both directions",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Directory for model output. Should be a non-empty string or \
            missing.",
    )
    return parser.parse_args()

def setup_logging(model: str, run_id: str) -> logging.Logger:
    """Set up the logger with a unique run ID."""
    log_dir = Path("logs") / model / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "get_embeddings.log"
    logger = setup_logger("get_embeddings", log_file)

    # Logging parameters
    for key, value in locals().items():
        logger.info(f"{key}: {value}")

    return logger


def find_missing_embeddings(
    protein_names: List[str], output_dir: Path, file_extension: str = ".npy"
) -> Optional[List[str]]:
    """
    Finds missing embeddings for the given input data and model name.

    Args:
        protein_names (List[str]): List of protein names for which
            embeddings are needed.
        output_dir (Path): Path to the directory containing embeddings.
        file_extension (str, optional): File extension for embeddings.  \
            Defaults to ".pt".

    Returns:
        Optional[List[str]]: List of missing protein names, or
        None if no missing embeddings.
    """

    # Ensure the output directory exists
    output_path = Path(output_dir)
    if not output_path.exists():
        return protein_names  # all embeddings are missing

    # Get the list of existing embeddings (strip file extension)
    existing_files = {
        file.stem for file in output_path.glob(f"*{file_extension}")
    }

    # Identify missing proteins
    missing_proteins = [
        name for name in protein_names if name not in list(existing_files)
    ]
    return missing_proteins


def get_sequences_from_uniprot(uniprot_ids):
    """
    Retrieves protein sequences from UniProt for the given list of UniProt IDs.

    Args:
        uniprot_ids (list): List of UniProt IDs (e.g., ['P12345', 'Q67890']).

    Returns:
        dict: A dictionary where keys are UniProt IDs and values are
        protein sequences.
    """
    base_url = "https://rest.uniprot.org/uniprotkb"
    sequences = {}

    for uniprot_id in tqdm(
        uniprot_ids, desc="Fetching UniProt Sequences", unit="protein"
    ):
        # Construct the URL for the FASTA request
        url = f"{base_url}/{uniprot_id}.fasta"
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            fasta_content = response.text
            # Extract the sequence from the FASTA content
            sequence = "".join(
                fasta_content.splitlines()[1:]
            )  # Skip the header line
            sequences[uniprot_id] = sequence
        else:
            print(
                f"Failed to retrieve sequence for UniProt ID {uniprot_id}. \
                    HTTP Status: {response.status_code}"
            )

    return sequences


def save_missing_proteins_fasta(
    sequences: Dict[str, str], run_id: str, logger: logging.Logger
) -> None:
    output_path = Path("data") / run_id
    output_path.mkdir(parents=True, exist_ok=True)
    fasta_path = output_path / "uniprot_sequences.fasta"
    write_fasta(sequences=sequences, file_path=fasta_path)
    logger.info(f"Missing protein sequences saved to {fasta_path}")


def validate_parameters(
    input: Union[str, List[str]],
    model: str,
    output_dir: str,
    max_seq_length: Optional[int],
    chunk_overlap: int,
    layer: int,
    no_gpu: bool,
    dev: bool,
    model_dir: str,
    bidrectional: bool = True,
) -> None:
    """
    Validates the parameters for the script.

    Args:
        input (Union[str, List[str]]): Input data (list of protein names \
            or path to a FASTA file).
        model (str): Model ID for embedding generation.
        output_dir (str): Directory to save embeddings.
        max_seq_length (Optional[int]): Maximum sequence length, can be None \
            or a non-negative integer.
        chunk_overlap (int): Overlap between chunks, must be a non-negative \
            integer.
        bidrectional (bool): Whether to chop sequences from both directions.
        layer (ing): Layer index to extract embeddings from.
        no_gpu (bool): Whether to disable GPU usage (must be a boolean).
        model_dir (str): Directory for model output. Should be a non-empty \
            string or missing.
        dev (bool): Whether to use development mode (must be a boolean).

    Raises:
        ValueError: If any of the parameters are invalid.
    """
    # Validate input
    if not input:
        raise ValueError("Input data is empty.")
    if isinstance(input, list) and len(input) != len(set(input)):
        raise ValueError("Input data contains duplicate entries.")
    if not isinstance(input, (list, str)):
        raise ValueError(
            "Input must be a list of protein names or a string pointing \
                to a FASTA file."
        )
    if isinstance(input, str) and not Path(input).is_file():
        raise ValueError(f"Input file does not exist: {input}")

    # Validate model
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Model ID must be a non-empty string.")
    EmbeddingModelMetadataHandler().validate_model_id(model)

    # Validate output_dir
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("Output directory must be a non-empty string.")
    if not os.path.isdir(output_dir):
        print(f"The directory '{output_dir}' does not exist. Creating it...")
        os.makedirs(output_dir)

    # Validate max_seq_length
    if max_seq_length is not None and (
        not isinstance(max_seq_length, int) or max_seq_length < 0
    ):
        raise ValueError(
            f"Invalid max_seq_length: {max_seq_length}. \
                It must be None or a non-negative integer."
        )
    # check max_seq_length is smaller or equal to the max_seq_length
    # of the chosen model
    if max_seq_length:
        max_seq_length_for_model = (
            EmbeddingModelMetadataHandler().get_max_seq_lengtg_per_model(
                model_name=model
            )
        )
        if max_seq_length > max_seq_length_for_model:
            raise ValueError(
                f"Invalid max_seq_length: {max_seq_length}.\
                    The maximum sequence length for model {model}\
                        is {max_seq_length_for_model}."
            )

    # Validate chunk_overlap
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError(
            f"Invalid chunk_overlap: {chunk_overlap}. \
                It must be a non-negative integer."
        )
    # chunk overlap should be less than max_seq_length
    if max_seq_length and chunk_overlap > max_seq_length:
        raise ValueError(
            "The chunk_overlap parameter must be less than max_seq_length."
        )

    # Validate layer
    if layer != -1:
        if not isinstance(layer, int) or layer < 0:
            raise ValueError(
                f"Invalid layer: {layer}. It must be -1 or a non-negative \
                    integer."
            )
    EmbeddingModelMetadataHandler().validate_repr_layers(
        model_id=model, repr_layers=[layer]
    )

    # Validate no_gpu
    if not isinstance(no_gpu, bool):
        raise ValueError("The no_gpu parameter must be a boolean.")

    # Validate dev
    if not isinstance(dev, bool):
        raise ValueError("The dev parameter must be a boolean.")

    # validate model_dir
    if model_dir is not None and (
        not isinstance(output_dir, str) or not output_dir.strip()
    ):
        raise ValueError("Output directory must be a non-empty string.")
    if not os.path.isdir(model_dir):
        print(f"The directory '{model_dir} does not exist. Creating it ...")
        os.makedirs(model_dir)

    # Validate bidrectional
    if not isinstance(bidrectional, bool):
        raise ValueError("The bidrectional parameter must be a boolean.")


def chop_sequences(
    sequence: str, max_length: int, overlap: int, bidirectional: bool = True
) -> List[str]:
    """
    i

    Args:
        sequence (str): sequence to be chopped
        max_length (int): max length of sequence for each chunk
        overlap (int): length of overlap between chunks
        bidirectional (book): whether to chop the sequence from both \
            directions (default: True) or only from the left

    Returns:
        List[str]: _description_
    """
    if len(sequence) <= max_length:
        return [sequence]

    n = len(sequence)

    chunks = []

    if not bidirectional:
        start = 0
        end = max_length
        while end < n:
            chunks.append(sequence[start:end])
            start = end - overlap
            end = start + max_length
    else:

        left_start = 0
        left_end = max_length
        chunks.append(sequence[left_start:left_end])

        right_start = n - max_length
        right_end = n
        chunks.append(sequence[right_start:right_end])

        while left_end - right_start <= overlap:
            left_start = left_end - overlap
            left_end = left_start + max_length
            chunks.append(sequence[left_start:left_end])
            if left_end - right_start > overlap:
                break

            right_end = right_start + overlap
            right_start = right_end - max_length
            chunks.insert(0, sequence[right_start:right_end])

        return chunks


def categorise_proteins_by_length(
    protein_sequences: Dict[str, str], max_seq_length: int
):
    """
    Categorizes proteins into two groups based on sequence length.

    Args:
        protein_sequences (Dict[str, str]): Dictionary of protein names and \
            sequences.
        max_seq_length (int): Maximum sequence length.

    Returns:
        Tuple[List[str], List[str]]: Tuple containing two lists of protein \
            names - one for proteins longer than max_seq_length and one for \
            proteins shorter than or equal to max_seq_length.
    """
    long_proteins = []
    short_proteins = []
    for protein, sequence in protein_sequences.items():
        if len(sequence) > max_seq_length:
            long_proteins.append(protein)
        else:
            short_proteins.append(protein)
    return long_proteins, short_proteins


def handle_short_proteins(
    short_proteins, output_dir, chopped_output_dir, logger
):
    """Handles short proteins by copying embeddings if already available."""
    proteins_to_remove = []

    for protein in short_proteins:
        if not find_missing_embeddings([protein], output_dir):
            src = output_dir / f"{protein}.npy"
            dst = chopped_output_dir / f"{protein}.npy"
            os.system(f"cp {src} {dst}")
            logger.info(f"Copied embedding for {protein} from {src} to {dst}")
            proteins_to_remove.append(protein)

    return proteins_to_remove


def map_protein_to_files(
    directory: Path, protein_names: List[str], extension: str = "npy"
) -> Dict[str, List[str]]:
    """
    Maps protein names to the files in a directory that match their
    name pattern.

    Args:
        directory (Path): The directory to search for files.
        protein_names (List[str]): A list of protein names to match.
        extension (str, optional): The file extension to filter by.
            Defaults to "npy".

    Returns:
        Dict[str, List[str]]: Dictionary where keys are protein names and
        values are lists of file names.
    """
    # Ensure the directory is a Path object
    directory = Path(directory)

    # Ensure the extension does not include a dot
    extension = extension.lstrip(".")

    # Initialize the mapping dictionary
    protein_file_map = {protein: [] for protein in protein_names}

    # Iterate through files in the directory
    for file in directory.glob(f"*.{extension}"):
        for protein in protein_names:
            if (
                file.stem.startswith(protein + "_")
                and file.stem[len(protein) + 1 :].isdigit()
            ):
                protein_file_map[protein].append(file.name)

    return protein_file_map


def validate_chopped_protein_file_names(
    protein_file_map: Dict[str, List[str]],
    n_chunks_per_protein: Dict[str, int],
    extension: str = "npy",
) -> Tuple[bool, List[str]]:
    """
    Validates the files for each protein based on two criteria:
    a) The number of files per protein matches the number of chunks for that
        protein.
    b) File indices are consecutive starting from 0 to (number of chunks - 1).

    Args:
        protein_file_map (Dict[str, List[str]]): Mapping of protein names to
            their file names.
        n_chunks_per_protein (Dict[str, int]): Mapping of protein names to the
            number of expected chunks.
        extension (str, optional): The file extension of the files.
            Defaults to "npy".

    Returns:
        Tuple[bool, List[str]]: A tuple containing:
            - A boolean indicating whether all checks passed.
            - A list of error messages, if any.
    """
    errors = []
    all_valid = True

    for protein, files in protein_file_map.items():
        # ensure the extension does not include a leading dot
        extension = extension.lstrip(".")

        # a) ensure number of files matches the expected number of chunks
        n_chunks_expected = n_chunks_per_protein[protein]
        if len(files) != n_chunks_expected:
            all_valid = False
            errors.append(
                f"Protein '{protein}' has {len(files)} files, \
                    but {n_chunks_expected} chunks are expected."
            )

        # b) check for consecutive indices starting from 0
        indices = []
        for file in files:
            try:
                index = int(file.split(f".{extension}")[0].split("_")[-1])
                indices.append(index)
            except ValueError:
                all_valid = False
                errors.append(
                    f"Invalid file name format for '{file}' \
                        in protein '{protein}'."
                )
        if indices:
            indices.sort()
            if indices != list(range(n_chunks_expected)):
                all_valid = False
                errors.append(
                    f"Protein '{protein}' has\
                        non-consecutive indices: {indices}.\
                        Expected: {list(range(n_chunks_expected))}"
                )

    return all_valid, errors


def aggregate_protein_embeddings(
    protein_file_map: Dict[str, List[str]],
    output_dir: Path,
    extension: str = "npy",
    remove_chunked_embeddings: bool = True,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Aggregates embeddings for each protein by averaging chunk embeddings.

    Args:
        protein_file_map (Dict[str, List[str]]): Mapping of protein names
            to their chunk file names.
        output_dir (Path): Directory to save the aggregated embeddings.
        extension (str, optional): File extension for the aggregated
            embeddings (default = "npy").
        remove_chunks (bool, optional): Whether to remove individual
            embedding files for the chunks after aggregation (default: True)
        logger (Optional[logging.Logger], optional): Logger for logging \
            messages (default: None).


    Raises:
        ValueError: If there are issues loading, validating, or processing \
            the chunk embeddings.
    """

    for protein, chunk_files in protein_file_map.items():
        embeddings = []
        embedding_shape = None

        # load all chunk embeddings for the protein
        for chunk_file in chunk_files:
            try:
                embedding = np.load(output_dir / chunk_file)
                if np.all(embedding == 0):
                    raise ValueError(
                        f"Embedding in {chunk_file} is all zeros."
                    )
                if embedding_shape is None:
                    embedding_shape = embedding.shape
                elif embedding.shape != embedding_shape:
                    raise ValueError(
                        f"Shape mismatch:\
                            {chunk_file} has shape {embedding.shape}, "
                        f"expected {embedding_shape}."
                    )
                embeddings.append(embedding)
            except Exception as e:
                raise ValueError(
                    f"Failed to load embedding from {chunk_file}: {e}"
                )

        # compute the mean acorss all chunks
        aggregated_embedding = np.mean(embeddings, axis=0)

        # save the aggregated embedding
        output_file = output_dir / f"{protein}.{extension}"
        if extension == "npy":
            np.save(output_file, aggregated_embedding)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")

        # optionally remove the chunk files
        if remove_chunked_embeddings:
            for chunk_file in chunk_files:
                try:
                    Path(output_dir / chunk_file).unlink()
                except Exception as e:
                    raise ValueError(
                        f"Failed to delete chunk file {chunk_file}: {e}"
                    )
        if logger:
            logger.info(
                f"Aggregated embedding for {protein} \
                saved to {output_file}"
            )


def get_embeddings(
    input: Union[str, List[str]],
    model: str,
    output_dir: str,
    run_id: Optional[str] = None,
    no_gpu: bool = False,
    dev: bool = False,
    layer: int = -1,
    output_format: str = "npy",
    max_seq_length: Optional[int] = None,
    chunk_overlap: Optional[int] = 0,
    logger: Optional[logging.Logger] = None,
    model_dir: Optional[str] = None,
    bidirectional: bool = True,
    per_protein: bool = True,
    **kwargs,
) -> None:
    """
    Retrieves embeddings for a given list of proteins or a FASTA file.

    Args:
        input (Union[List[str], str]): List of protein names or path to a \
            FASTA file.
        model (str): Model ID to use for embedding generation.
        output_dir (str): Directory to save embeddings.
        run_id (Optional[str], default=None): Unique identifier for the run.
        no_gpu (bool, default=False): If True, use CPU instead of GPU.
        dev (bool, default=False): If True, process only a subset of proteins.
        max_seq_length (Optional[int], default=None): Maximum sequence length.
            Sequences longer than this will be chopped.
        chunk_overlap (Optional[int], default=0): Overlap size between chunks.
        layer (int, default=-1): Layer index to extract embeddings from.
        logger (Optional[logging.Logger], default=None): Logger for the \
            function.
        output_format (Optional[str], default="npy"): File format for \
            the embeddings. Default is npy file.
        bidirectional (bool, default=True): Whether to chop sequences from \
            both directions.
        per_protein: bool, default=True: Whether to aggregate embeddings \
            per protein.
        **kwargs: Additional arguments for the embedding handler.

    Raises:
        ValueError: If input is invalid or embeddings cannot be generated.
    """

    # if run_id is not provided, generate a new one
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logger or setup_logging(model, run_id)

    # logging of parameters
    for key, value in locals().items():
        logger.info(f"{key}: {value}")

    validate_parameters(
        input=input,
        model=model,
        output_dir=output_dir,
        max_seq_length=max_seq_length,
        chunk_overlap=chunk_overlap,
        layer=layer,
        no_gpu=no_gpu,
        dev=dev,
        model_dir=model_dir,
        bidrectional=bidirectional,
    )

    if layer == -1:
        layer = EmbeddingModelMetadataHandler().get_last_layer_per_model_id(
            model
        )

    # Parse input data to extract protein names
    if isinstance(input, list):
        protein_names = input
    elif isinstance(input, str) and Path(input).suffix.lower() == ".fasta":
        protein_names = read_fasta_names(input)
    else:
        raise ValueError(
            "Input data must be a list of protein names or a valid \
                FASTA file path."
        )

    # check for embeddings to be retrieved
    # prepare output directory
    output_dir = Path(output_dir) / model / f"layer_{layer}"
    if max_seq_length:
        chopped_output_dir = (
            output_dir / f"chopped_{max_seq_length}_overlap_{chunk_overlap}"
        )
    else:
        chopped_output_dir = output_dir

    if not chopped_output_dir.exists():
        chopped_output_dir.mkdir(parents=True, exist_ok=True)

    missing_embeddings = find_missing_embeddings(
        protein_names, chopped_output_dir
    )

    if not missing_embeddings:
        logger.info("All embeddings are already retrieved.")
        return

    logger.info(
        f"Retrieving embeddings for {len(missing_embeddings)} proteins"
    )

    if dev:
        # pick only the first 10 proteins
        missing_embeddings = missing_embeddings[:10]

    # Retrieve sequences for missing proteins if input is a FASTA file
    missing_protein_sequences = {}
    if isinstance(input, str) and Path(input).suffix.lower() == ".fasta":
        sequences = read_fasta(file_path=input)
        missing_protein_sequences = {
            protein: sequences[protein] for protein in missing_embeddings
        }
    elif isinstance(input, list):
        missing_protein_sequences = get_sequences_from_uniprot(
            missing_embeddings
        )
        save_missing_proteins_fasta(missing_protein_sequences, run_id, logger)

    # chop sequences if needed
    if max_seq_length:
        # test how many sequences are too short to be chopped
        long_proteins, short_proteins = categorise_proteins_by_length(
            missing_protein_sequences, max_seq_length
        )

        # Handle short proteins
        proteins_to_remove = handle_short_proteins(
            short_proteins, output_dir, chopped_output_dir, logger
        )
        short_proteins = [
            protein
            for protein in short_proteins
            if protein not in proteins_to_remove
        ]

        logger.info(
            f"Copied embeddings for {len(proteins_to_remove)} short proteins."
        )

        chopped_sequences = {}
        n_chunks_per_protein = {}
        for protein in long_proteins:
            sequence = missing_protein_sequences[protein]
            chunks = chop_sequences(
                sequence,
                max_length=max_seq_length,
                overlap=chunk_overlap,
                bidirectional=bidirectional,
            )
            logger.info(f"Chopped {protein} into {len(chunks)} sequences.")
            n_chunks_per_protein[protein] = len(chunks)
            for i, chunk in enumerate(chunks):
                chopped_sequences[f"{protein}_{i}"] = chunk

        # Add short proteins directly to chopped_sequences
        chopped_sequences.update(
            {
                protein: missing_protein_sequences[protein]
                for protein in short_proteins
            }
        )

        # Check for missing embeddings again
        missing_embeddings = find_missing_embeddings(
            list(chopped_sequences), chopped_output_dir
        )
        chopped_sequences = {
            protein: sequence
            for protein, sequence in chopped_sequences.items()
            if protein in missing_embeddings
        }
    else:
        chopped_sequences = missing_protein_sequences

    if chopped_sequences != {}:
        # get embeddings for missing proteins
        embedding_handler = select_embedding_handler(
            model_id=model, logger=logger, no_gpu=no_gpu
        )
        embedding_handler.get_embedding(
            protein_sequences=chopped_sequences,
            model_id=model,
            output_dir=chopped_output_dir,
            layer=layer,
            truncation_seq_length=max_seq_length,
            model_dir=model_dir,
            per_protein=per_protein,
            # add batch size parameter
            **kwargs,
        )

    # aggregate embeddings
    if max_seq_length and per_protein:
        logger.info(
            f"Aggregating embeddings for {len(long_proteins)} proteins."
        )
        # find all embeddings for this protein
        chopped_files_per_protein = map_protein_to_files(
            directory=chopped_output_dir,
            protein_names=long_proteins,
            extension=output_format,
        )
        # validate chunk names
        all_valid, errors = validate_chopped_protein_file_names(
            protein_file_map=chopped_files_per_protein,
            n_chunks_per_protein=n_chunks_per_protein,
            extension=output_format,
        )
        if not all_valid:
            error_message = (
                "Validation failed for protein file outputs:\n"
                + "\n".join(errors)
            )
            logger.error(error_message)
            raise ValueError(error_message)

        # aggregate embeddings for long_proteins
        aggregate_protein_embeddings(
            protein_file_map=chopped_files_per_protein,
            output_dir=chopped_output_dir,
            extension=output_format,
            remove_chunked_embeddings=True,
            logger=logger,
        )
    logger.info("All embeddings calculated.")


def main():
    args = parse_args()
    print(args)
    get_embeddings(**vars(args))


if __name__ == "__main__":
    main()
