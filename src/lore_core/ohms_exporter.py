import uuid
from lxml import etree
import datetime
from pathlib import Path
from models.transcript import Transcript


def _format_vtt_time(seconds: float) -> str:
    """Format float seconds to strictly HH:MM:SS.mmm format"""
    td = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    ms = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _prune_empty_elements(element):
    """Recursively remove empty tags from the tree"""
    for child in list(element):
        _prune_empty_elements(child)
        if not child.text and not len(child) and not child.attrib:
            element.remove(child)
    if element.text and not element.text.strip():
        element.text = None


class OhmsExporter:
    """
    Exports a Transcript object to OHMS XML 6.0 format.
    """

    OHMS_NS = "https://www.weareavp.com/nunncenter/ohms"
    XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
    SCHEMA_LOC = "https://www.weareavp.com/nunncenter/ohms/ohms.xsd"

    @classmethod
    def export(cls, transcript: Transcript, metadata: dict, output_path: Path) -> None:
        nsmap = {None: cls.OHMS_NS, "xsi": cls.XSI_NS}

        root = etree.Element(f"{{{cls.OHMS_NS}}}ROOT", nsmap=nsmap)
        root.set(f"{{{cls.XSI_NS}}}schemaLocation", cls.SCHEMA_LOC)

        # Generate IDs/dates
        record_id = metadata.get("record_id") or str(uuid.uuid4())[:8]
        dt = datetime.datetime.now().strftime("%Y-%m-%d")

        record = etree.SubElement(root, f"{{{cls.OHMS_NS}}}record", id=record_id, dt=dt)

        # Version
        version = etree.SubElement(record, f"{{{cls.OHMS_NS}}}version")
        version.text = "6.0"

        # Date
        etree.SubElement(
            record, f"{{{cls.OHMS_NS}}}date", value=dt, format="yyyy-mm-dd"
        )

        # Required sequence
        etree.SubElement(record, f"{{{cls.OHMS_NS}}}title").text = metadata.get(
            "title", "Untitled"
        )
        etree.SubElement(record, f"{{{cls.OHMS_NS}}}repository").text = metadata.get(
            "repository", ""
        )
        etree.SubElement(record, f"{{{cls.OHMS_NS}}}interviewee").text = metadata.get(
            "interviewee", ""
        )
        etree.SubElement(record, f"{{{cls.OHMS_NS}}}interviewer").text = metadata.get(
            "interviewer", ""
        )
        etree.SubElement(record, f"{{{cls.OHMS_NS}}}format").text = metadata.get(
            "format", "audio/mp3"
        )
        etree.SubElement(record, f"{{{cls.OHMS_NS}}}media_url").text = metadata.get(
            "media_url", ""
        )

        abstract_text = metadata.get("abstract", "")
        if abstract_text:
            etree.SubElement(
                record, f"{{{cls.OHMS_NS}}}description"
            ).text = abstract_text

        etree.SubElement(
            record, f"{{{cls.OHMS_NS}}}language"
        ).text = transcript.metadata.language
        if transcript.metadata.target_language:
            etree.SubElement(
                record, f"{{{cls.OHMS_NS}}}language_for_translation"
            ).text = transcript.metadata.target_language
            etree.SubElement(
                record, f"{{{cls.OHMS_NS}}}include_translation"
            ).text = "true"

        # Collect NER entities from segments, separated by label type
        keywords_set = set()
        subjects_set = set()
        for seg in transcript.segments:
            for ent in seg.entities:
                if ent.label in ("person", "organization", "location"):
                    keywords_set.add(ent.text)
            for tag in seg.tags:
                if tag.preferred_term:
                    subjects_set.add(tag.preferred_term)

        keywords_str = "; ".join(sorted(keywords_set))
        subjects_str = "; ".join(sorted(subjects_set))

        index_node = etree.SubElement(record, f"{{{cls.OHMS_NS}}}index")
        if keywords_str or subjects_str:
            point = etree.SubElement(index_node, f"{{{cls.OHMS_NS}}}point")
            etree.SubElement(point, f"{{{cls.OHMS_NS}}}time").text = "0"
            etree.SubElement(
                point, f"{{{cls.OHMS_NS}}}title"
            ).text = "Auto-generated Entities"
            if keywords_str:
                etree.SubElement(point, f"{{{cls.OHMS_NS}}}keywords").text = keywords_str
            if subjects_str:
                etree.SubElement(point, f"{{{cls.OHMS_NS}}}subjects").text = subjects_str

        # Generate VTT Transcript
        vtt_transcript = etree.SubElement(record, f"{{{cls.OHMS_NS}}}vtt_transcript")

        vtt_content = "WEBVTT\n\n"
        for seg in transcript.segments:
            start = _format_vtt_time(seg.start_ms / 1000.0)
            end = _format_vtt_time(seg.end_ms / 1000.0)
            seg_text = seg.text
            # Append [overlap] if segment intersects any OverlapRegion
            for region in transcript.overlap_regions:
                if region.start_ms < seg.end_ms and region.end_ms > seg.start_ms:
                    seg_text += " [overlap]"
                    break
            vtt_content += f"{start} --> {end}\n"
            if seg.speaker_label:
                vtt_content += f"<v {seg.speaker_label}>{seg_text}\n\n"
            else:
                vtt_content += f"{seg_text}\n\n"

        vtt_transcript.text = etree.CDATA(vtt_content.strip())

        if transcript.metadata.target_language and any(
            seg.translation for seg in transcript.segments
        ):
            vtt_alt = etree.SubElement(record, f"{{{cls.OHMS_NS}}}vtt_transcript_alt")
            vtt_alt_content = "WEBVTT\n\n"
            for seg in transcript.segments:
                start = _format_vtt_time(seg.start_ms / 1000.0)
                end = _format_vtt_time(seg.end_ms / 1000.0)
                text = seg.translation if seg.translation else seg.text
                # Append [overlap] if segment intersects any OverlapRegion
                for region in transcript.overlap_regions:
                    if region.start_ms < seg.end_ms and region.end_ms > seg.start_ms:
                        text += " [overlap]"
                        break
                vtt_alt_content += f"{start} --> {end}\n"
                if seg.speaker_label:
                    vtt_alt_content += f"<v {seg.speaker_label}>{text}\n\n"
                else:
                    vtt_alt_content += f"{text}\n\n"

            vtt_alt.text = etree.CDATA(vtt_alt_content.strip())

        # Rights
        etree.SubElement(record, f"{{{cls.OHMS_NS}}}rights").text = metadata.get(
            "rights", ""
        )

        # Prune empty
        _prune_empty_elements(root)

        xml_bytes = etree.tostring(
            root, pretty_print=True, xml_declaration=True, encoding="UTF-8"
        )

        with open(output_path, "wb") as f:
            f.write(xml_bytes)
