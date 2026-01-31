"""
Input validation utilities for the RAG system.
Provides file type validation, size limits, and path security checks.
"""

import os
from pathlib import Path
from typing import BinaryIO

from config import settings
from logger import logger


class ValidationError(Exception):
    """Custom exception for validation failures."""

    pass


def validate_file_size(file_path: str | None = None, file_obj: BinaryIO | None = None) -> None:
    """
    Validate that a file doesn't exceed the maximum allowed size.

    Args:
        file_path: Path to the file to validate.
        file_obj: File-like object to validate (for Streamlit uploads).

    Raises:
        ValidationError: If file exceeds size limit.
    """
    max_size = settings.max_file_size_bytes

    if file_path:
        file_size = os.path.getsize(file_path)
    elif file_obj:
        # Get size from file object
        current_pos = file_obj.tell()
        file_obj.seek(0, 2)  # Seek to end
        file_size = file_obj.tell()
        file_obj.seek(current_pos)  # Reset position
    else:
        raise ValidationError("Either file_path or file_obj must be provided")

    if file_size > max_size:
        size_mb = file_size / (1024 * 1024)
        logger.warning(
            f"File size validation failed: {size_mb:.2f}MB exceeds {settings.max_file_size_mb}MB limit"
        )
        raise ValidationError(
            f"File size ({size_mb:.2f}MB) exceeds maximum allowed size ({settings.max_file_size_mb}MB)"
        )

    logger.debug(f"File size validation passed: {file_size / (1024 * 1024):.2f}MB")


def validate_file_type(file_path: str | None = None, file_bytes: bytes | None = None) -> None:
    """
    Validate file type using magic bytes (not just extension).

    Args:
        file_path: Path to the file to validate.
        file_bytes: First bytes of the file for magic number check.

    Raises:
        ValidationError: If file type is not allowed.
    """
    # PDF magic bytes: %PDF
    PDF_MAGIC = b"%PDF"

    if file_path:
        with open(file_path, "rb") as f:
            header = f.read(4)
    elif file_bytes:
        header = file_bytes[:4]
    else:
        raise ValidationError("Either file_path or file_bytes must be provided")

    if not header.startswith(PDF_MAGIC):
        logger.warning(f"File type validation failed: not a valid PDF file")
        raise ValidationError("Invalid file type. Only PDF files are allowed.")

    logger.debug("File type validation passed: valid PDF")


def validate_path_security(file_path: str, base_dir: str) -> str:
    """
    Validate that a file path doesn't escape the base directory (path traversal protection).

    Args:
        file_path: The file path to validate.
        base_dir: The base directory files should be contained in.

    Returns:
        The resolved absolute path if valid.

    Raises:
        ValidationError: If path traversal is detected.
    """
    # Resolve to absolute paths
    base_path = Path(base_dir).resolve()
    target_path = Path(os.path.join(base_dir, file_path)).resolve()

    # Check if target is within base directory
    try:
        target_path.relative_to(base_path)
    except ValueError:
        logger.error(f"Path traversal attempt detected: {file_path}")
        raise ValidationError("Invalid file path: path traversal not allowed")

    return str(target_path)


def validate_filename(filename: str) -> str:
    """
    Sanitize and validate a filename.

    Args:
        filename: The filename to validate.

    Returns:
        Sanitized filename.

    Raises:
        ValidationError: If filename is invalid.
    """
    if not filename:
        raise ValidationError("Filename cannot be empty")

    # Remove any path components
    filename = os.path.basename(filename)

    # Check for dangerous characters
    dangerous_chars = ["<", ">", ":", '"', "/", "\\", "|", "?", "*", "\x00"]
    for char in dangerous_chars:
        if char in filename:
            logger.warning(f"Invalid character '{char}' in filename: {filename}")
            raise ValidationError(f"Invalid character in filename: {char}")

    # Check extension
    if not filename.lower().endswith(".pdf"):
        raise ValidationError("Only PDF files are allowed")

    logger.debug(f"Filename validation passed: {filename}")
    return filename


def validate_file(
    file_path: str | None = None,
    file_obj: BinaryIO | None = None,
    file_bytes: bytes | None = None,
    filename: str | None = None,
) -> None:
    """
    Perform all file validations.

    Args:
        file_path: Path to the file.
        file_obj: File-like object (for uploads).
        file_bytes: Raw file bytes.
        filename: Original filename.

    Raises:
        ValidationError: If any validation fails.
    """
    # Validate filename if provided
    if filename:
        validate_filename(filename)

    # Validate file size
    validate_file_size(file_path=file_path, file_obj=file_obj)

    # Validate file type
    if file_path:
        validate_file_type(file_path=file_path)
    elif file_bytes:
        validate_file_type(file_bytes=file_bytes)
    elif file_obj:
        current_pos = file_obj.tell()
        file_bytes = file_obj.read(4)
        file_obj.seek(current_pos)
        validate_file_type(file_bytes=file_bytes)

    logger.info("All file validations passed")
