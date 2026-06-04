import os
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from models.transcript import Transcript
from lore_core.ohms_exporter import OhmsExporter

class BagItPackager:
    """
    Creates a BagIt (RFC 8493) compliant package for the transcript and audio.
    """
    def __init__(self):
        self.version = "1.0"
        self.encoding = "UTF-8"
        
    def _hash_file(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def create_bag(self, audio_path: Path, transcript: Transcript, output_dir: Path, project_id: str):
        """
        Generates a BagIt structure at output_dir.
        """
        output_dir = Path(output_dir)
        bag_dir = output_dir / f"lore_bag_{project_id}"
        
        # Ensure we start clean
        if bag_dir.exists():
            shutil.rmtree(bag_dir)
            
        bag_dir.mkdir(parents=True)
        data_dir = bag_dir / "data"
        data_dir.mkdir()
        
        # 1. Copy audio payload
        bag_audio_path = data_dir / audio_path.name
        shutil.copy2(audio_path, bag_audio_path)
        
        # 2. Export OHMS XML payload
        bag_xml_path = data_dir / "transcript.xml"
        OhmsExporter.export(transcript, {"title": project_id}, bag_xml_path)
            
        # 3. Create bagit.txt
        with open(bag_dir / "bagit.txt", 'w', encoding='utf-8') as f:
            f.write(f"BagIt-Version: {self.version}\n")
            f.write(f"Tag-File-Character-Encoding: {self.encoding}\n")
            
        # 4. Create bag-info.txt
        with open(bag_dir / "bag-info.txt", 'w', encoding='utf-8') as f:
            f.write("Source-Organization: Lore Oral History App\n")
            f.write(f"Bagging-Date: {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write(f"External-Identifier: {project_id}\n")
            # Calculate total payload size and oxum
            audio_size = bag_audio_path.stat().st_size
            xml_size = bag_xml_path.stat().st_size
            total_size = audio_size + xml_size
            f.write(f"Payload-Oxum: {total_size}.2\n")
            f.write(f"Bag-Size: {total_size / (1024*1024):.2f} MB\n")
            
        # 5. Create payload manifest (manifest-sha256.txt)
        with open(bag_dir / "manifest-sha256.txt", 'w', encoding='utf-8') as f:
            audio_hash = self._hash_file(bag_audio_path)
            xml_hash = self._hash_file(bag_xml_path)
            f.write(f"{audio_hash} data/{bag_audio_path.name}\n")
            f.write(f"{xml_hash} data/transcript.xml\n")
            
        # 6. Create tag manifest (tagmanifest-sha256.txt)
        with open(bag_dir / "tagmanifest-sha256.txt", 'w', encoding='utf-8') as f:
            bagit_hash = self._hash_file(bag_dir / "bagit.txt")
            baginfo_hash = self._hash_file(bag_dir / "bag-info.txt")
            manifest_hash = self._hash_file(bag_dir / "manifest-sha256.txt")
            f.write(f"{bagit_hash} bagit.txt\n")
            f.write(f"{baginfo_hash} bag-info.txt\n")
            f.write(f"{manifest_hash} manifest-sha256.txt\n")
            
        return bag_dir
