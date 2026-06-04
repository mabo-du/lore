import os
from pathlib import Path
from lore_core.bagit_exporter import BagItPackager

def test_bagit_creation(tmp_path, dummy_audio_file, dummy_transcript):
    """
    Test that the BagIt packager correctly creates an RFC 8493 compliant package
    without throwing errors.
    """
    output_dir = tmp_path / "export"
    packager = BagItPackager()
    
    bag_dir = packager.create_bag(
        audio_path=dummy_audio_file,
        transcript=dummy_transcript,
        output_dir=output_dir,
        project_id="test_project_123"
    )
    
    # Verify structure
    assert bag_dir.exists()
    assert (bag_dir / "data").exists()
    assert (bag_dir / "data" / dummy_audio_file.name).exists()
    assert (bag_dir / "data" / "transcript.xml").exists()
    
    # Verify BagIt mandated files
    assert (bag_dir / "bagit.txt").exists()
    assert (bag_dir / "bag-info.txt").exists()
    assert (bag_dir / "manifest-sha256.txt").exists()
    assert (bag_dir / "tagmanifest-sha256.txt").exists()
    
    # Verify content of bagit.txt
    bagit_content = (bag_dir / "bagit.txt").read_text()
    assert "BagIt-Version: 1.0" in bagit_content
    
    # Verify manifest contains our files
    manifest_content = (bag_dir / "manifest-sha256.txt").read_text()
    assert dummy_audio_file.name in manifest_content
    assert "transcript.xml" in manifest_content
