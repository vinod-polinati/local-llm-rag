"""
Unit tests for the validators module.
Tests file validation, size limits, and path security.
"""

import os
import tempfile
import pytest

from validators import (
    validate_file_size,
    validate_file_type,
    validate_path_security,
    validate_filename,
    ValidationError,
)


class TestFileSizeValidation:
    """Tests for file size validation."""

    def test_valid_file_size(self, temp_dir):
        """Test that files under the limit pass validation."""
        # Create a small file
        file_path = os.path.join(temp_dir, "small.txt")
        with open(file_path, "w") as f:
            f.write("Small content")
        
        # Should not raise
        validate_file_size(file_path=file_path)

    def test_file_path_required(self):
        """Test that either file_path or file_obj is required."""
        with pytest.raises(ValidationError, match="must be provided"):
            validate_file_size()


class TestFileTypeValidation:
    """Tests for file type validation."""

    def test_valid_pdf_magic_bytes(self):
        """Test that valid PDF magic bytes pass."""
        pdf_magic = b"%PDF-1.4 some content here"
        # Should not raise
        validate_file_type(file_bytes=pdf_magic)

    def test_invalid_file_type_raises(self):
        """Test that non-PDF files raise ValidationError."""
        text_content = b"This is just plain text"
        
        with pytest.raises(ValidationError, match="PDF"):
            validate_file_type(file_bytes=text_content)

    def test_file_or_bytes_required(self):
        """Test that either file_path or file_bytes is required."""
        with pytest.raises(ValidationError, match="must be provided"):
            validate_file_type()


class TestPathSecurityValidation:
    """Tests for path traversal protection."""

    def test_normal_path_passes(self, temp_dir):
        """Test that normal paths pass validation."""
        result = validate_path_security("document.pdf", temp_dir)
        assert result is not None

    def test_path_traversal_blocked(self, temp_dir):
        """Test that path traversal attempts are blocked."""
        with pytest.raises(ValidationError, match="traversal"):
            validate_path_security("../../../etc/passwd", temp_dir)


class TestFilenameValidation:
    """Tests for filename validation."""

    def test_valid_filename(self):
        """Test that valid filenames pass."""
        result = validate_filename("document.pdf")
        assert result == "document.pdf"

    def test_empty_filename_raises(self):
        """Test that empty filenames raise error."""
        with pytest.raises(ValidationError, match="empty"):
            validate_filename("")

    def test_dangerous_chars_raise(self):
        """Test that dangerous characters raise error."""
        with pytest.raises(ValidationError, match="Invalid character"):
            validate_filename("file<name>.pdf")

    def test_non_pdf_raises(self):
        """Test that non-PDF extensions raise error."""
        with pytest.raises(ValidationError, match="PDF"):
            validate_filename("document.exe")
