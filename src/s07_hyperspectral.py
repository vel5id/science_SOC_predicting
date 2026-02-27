"""
Hyperspectral Data Investigation
=================================
Check availability of PRISMA, EnMAP, and LUCAS data for the study area.

Output: data/hyperspectral/availability_report.txt
"""
from __future__ import annotations

import requests

from . import config
from .db_utils import get_region_bbox
from .file_utils import should_skip_file


def check_prisma_availability(bbox: tuple[float, float, float, float]) -> dict:
    """
    Check PRISMA hyperspectral data availability.
    
    Note: PRISMA data access requires ESA account and manual search.
    This is a placeholder for automated checking.
    """
    return {
        "source": "PRISMA (ASI/ESA)",
        "available": False,
        "note": "Requires manual search at https://prisma.asi.it/",
        "coverage": "Limited global coverage, primarily Europe and Mediterranean",
    }


def check_enmap_availability(bbox: tuple[float, float, float, float]) -> dict:
    """
    Check EnMAP hyperspectral data availability.
    
    Note: EnMAP data access requires registration.
    This is a placeholder for automated checking.
    """
    return {
        "source": "EnMAP (DLR)",
        "available": False,
        "note": "Requires manual search at https://www.enmap.org/",
        "coverage": "Limited global coverage, mission started 2022",
    }


def check_lucas_availability() -> dict:
    """
    Check LUCAS soil spectral library availability.
    
    LUCAS is a point-based soil spectral library, not imagery.
    """
    return {
        "source": "LUCAS Soil Spectral Library (JRC)",
        "available": True,
        "note": "Available at https://esdac.jrc.ec.europa.eu/",
        "coverage": "EU countries only, point samples",
        "relevance": "Low - study area is Kazakhstan (outside EU)",
    }


def main() -> None:
    """Main execution: check hyperspectral data availability."""
    print("="*60)
    print("Hyperspectral Data Availability Investigation")
    print("="*60)
    
    print("\nFetching region bounding box from database...")
    bbox = get_region_bbox()
    print(f"  BBOX: {bbox}")
    
    config.HYPER_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if report already exists
    report_path = config.HYPER_DIR / "availability_report.txt"
    if should_skip_file(report_path):
        print(f"\n✓ Already exists: {report_path.name}")
        print("\n" + "="*60)
        print("Recommendation: Focus on S2/L8/S1 for this study area.")
        print("Hyperspectral data (PRISMA/EnMAP) unlikely to be available.")
        print("="*60)
        return
    
    print("\nChecking data sources...")
    
    # Check each source
    prisma = check_prisma_availability(bbox)
    enmap = check_enmap_availability(bbox)
    lucas = check_lucas_availability()
    
    sources = [prisma, enmap, lucas]
    
    # Print results
    for src in sources:
        print(f"\n{src['source']}:")
        print(f"  Available: {src['available']}")
        print(f"  Note: {src['note']}")
        if 'coverage' in src:
            print(f"  Coverage: {src['coverage']}")
        if 'relevance' in src:
            print(f"  Relevance: {src['relevance']}")
    
    # Save report
    with open(report_path, "w") as f:
        f.write("Hyperspectral Data Availability Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Study Area BBOX: {bbox}\n\n")
        
        for src in sources:
            f.write(f"{src['source']}\n")
            f.write(f"  Available: {src['available']}\n")
            f.write(f"  Note: {src['note']}\n")
            if 'coverage' in src:
                f.write(f"  Coverage: {src['coverage']}\n")
            if 'relevance' in src:
                f.write(f"  Relevance: {src['relevance']}\n")
            f.write("\n")
    
    print(f"\n✓ Report saved: {report_path.name}")
    
    print("\n" + "="*60)
    print("Recommendation: Focus on S2/L8/S1 for this study area.")
    print("Hyperspectral data (PRISMA/EnMAP) unlikely to be available.")
    print("="*60)


if __name__ == "__main__":
    main()
