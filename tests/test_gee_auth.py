"""Tests for gee_auth.py"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src import gee_auth


@patch("src.gee_auth.ee")
def test_authenticate_and_initialize_already_authenticated(mock_ee):
    """Test when GEE is already authenticated."""
    # Mock successful initialization
    mock_ee.Initialize.return_value = None
    
    gee_auth.authenticate_and_initialize()
    
    # Should call Initialize once
    mock_ee.Initialize.assert_called_once()
    # Should not call Authenticate
    mock_ee.Authenticate.assert_not_called()


@patch("src.gee_auth.ee")
def test_authenticate_and_initialize_needs_auth(mock_ee):
    """Test when GEE needs authentication."""
    # Mock Initialize to fail first time, succeed second time
    mock_ee.Initialize.side_effect = [Exception("Not authenticated"), None]
    mock_ee.Authenticate.return_value = None
    
    gee_auth.authenticate_and_initialize()
    
    # Should call Initialize twice (fail, then succeed)
    assert mock_ee.Initialize.call_count == 2
    # Should call Authenticate once
    mock_ee.Authenticate.assert_called_once()


@patch("src.gee_auth.ee")
def test_check_gee_ready_success(mock_ee):
    """Test GEE ready check when successful."""
    mock_ee.Initialize.return_value = None
    mock_number = MagicMock()
    mock_number.getInfo.return_value = 1
    mock_ee.Number.return_value = mock_number
    
    result = gee_auth.check_gee_ready()
    
    assert result is True
    mock_ee.Initialize.assert_called_once()
    mock_ee.Number.assert_called_once_with(1)


@patch("src.gee_auth.ee")
def test_check_gee_ready_failure(mock_ee):
    """Test GEE ready check when failed."""
    mock_ee.Initialize.side_effect = Exception("GEE not ready")
    
    result = gee_auth.check_gee_ready()
    
    assert result is False
