# **Deep Technical Analysis of the OHMS XML Schema and Oral History Deposit Ecosystem**

## **Executive Overview**

The landscape of digital oral history archiving is undergoing a profound structural and epistemological shift. Historically, institutional repositories relied upon static, text-based, word-for-word transcripts or analog indexing methodologies to provide access to time-based media. These traditional approaches often resulted in the audio or video recording remaining an undivided media file that could span multiple hours, rendering the primary source material cumbersome and inaccessible to researchers seeking specific conceptual insights.1 The advent of digital synchronization frameworks has fundamentally altered this paradigm. At the forefront of this shift is the Oral History Metadata Synchronizer (OHMS), an open-source, web-based system developed by the Louie B. Nunn Center for Oral History at the University of Kentucky Libraries.1 By closely intertwining the transcript and the recording, OHMS makes browsing oral histories interactive, enabling transcript sections and thematic index segments to display in precise synchronization with the media playback.1  
For a privacy-first, edge-computed transcription application such as Lore, which utilizes the faster-whisper machine learning model for automatic speech recognition, producing highly accurate, schema-compliant OHMS XML is an operational necessity. Lore’s primary differentiator lies in its ability to produce outputs in formats immediately acceptable to institutional repositories without relying on cloud-based processing.3 The generated output must interface flawlessly with an array of institutional content management systems (CMS)—including Omeka S, CONTENTdm, and Aviary—utilizing standardized metadata structures to guarantee long-term preservation and immediate accessibility.5  
This comprehensive research report details the technical specifications of the OHMS XML schema, maps the repository platform ecosystem, evaluates competing and complementary synchronization standards, summarizes the ethical metadata requirements dictated by the oral history community, and provides exhaustive implementation guidance for deploying robust Python-based XML serializers within the Lore application architecture.

## **1\. OHMS XML Schema: Technical Specification**

The OHMS XML architecture functions not as a proprietary, closed ecosystem, but as a flexible, interoperable metadata schema designed specifically to bridge the technical gap between static textual data and time-based media assets. The schema provides a unifying wrapper that houses both the presentation logic (the transcript) and the semantic logic (the index) within a single preservable archival asset.1

### **1.1 Current Schema Version and Namespace Architecture**

As of late 2023 through the 2025–2026 operational window, the OHMS schema ecosystem has transitioned significantly from its legacy iterations. Historically operating on OHMS XML version 5.2 and version 5.4, the schema advanced to OHMS XML 6.0 following the OHMS application's native integration into the Aviary platform in September 2023\.8 This integration fundamentally changed the underlying standards utilized for transcript synchronization.  
The defining namespace and the centralized location for the OHMS XML Schema Definition (XSD) are established as https://www.weareavp.com/nunncenter/ohms and https://www.weareavp.com/nunncenter/ohms/ohms.xsd, respectively, with older documentation referencing the legacy AVPreserve URLs (e.g., https://www.avpreserve.com/nunncenter/ohms/ohms.xsd).11 The transition from legacy versions to version 6.0 represents a crucial shift in how timecodes are mapped to textual data, transitioning from an integer-based seconds mapping to a granular float-based model.10  
Under legacy schema versions (up to v5.4), the \<transcript\> block utilized specific \<sync\> and \<sync\_alt\> tags.10 This legacy synchronization relied on a "minute-marker" paradigm. During the manual preparation of an oral history, an archivist utilizing the OHMS web application would listen for a low beep indicating the start of a minute, and a high beep indicating the end of a ten-second synchronization window, dropping a timestamp at regular intervals rather than mapping specific words or sentences to exact milliseconds.13  
OHMS XML 6.0 abandons this manual paradigm, introducing the \<vtt\_transcript\> and \<vtt\_transcript\_alt\> elements.10 This adoption of the WebVTT (Web Video Text Tracks) transcription standard allows for granular, continuous timecodes marking the exact start and end of specific segments or paragraphs.10 For an automated, edge-computed transcription application like Lore, which utilizes the faster-whisper model to generate word-level and segment-level timestamps natively, targeting the OHMS XML 6.0 schema is the optimal architectural path for serialization. This allows Lore to output high-precision machine learning data without down-sampling the timestamps to fit the legacy one-minute increment model.

### **1.2 Mandatory vs. Optional Elements and Validation Criteria**

To ensure successful validation against the XSD and flawless ingestion into a host institution's CMS, strict adherence to the schema’s element requirements is essential. Within the OHMS application’s internal database logic, an interview must contain at least three core metadata fields to be registered: the Title, the Media Format, and the Media Connection (which includes variants such as the Media URL, a Media Host ID, or an iFrame Embed Code).14  
When exporting to the XML file, the \<record\> block acts as a stable envelope that wraps the underlying media and textual metadata. The absence of the \<title\> element or the \<media\_url\> element will cause catastrophic rendering failures in the client-side OHMS Viewer, as the viewer application relies on these specific nodes to instantiate the HTML5 media player or interface with third-party streaming APIs like YouTube, Vimeo, or Kaltura.7  
Conversely, elements that provide extended contextual data—such as the \<user\_note\>, \<gps\>, \<keywords\>, and \<subjects\> fields—are strictly optional within the schema definition.9 The modern iterations of the OHMS Viewer (version 3.6.0 and higher) handle these empty optional fields gracefully. If a segment does not contain any subjects, the viewer logic suppresses the “Subjects” label entirely, presenting a clean interface to the end user.9 However, older versions of the OHMS Viewer experienced significant interface skewing and formatting anomalies due to empty fields.9 This highlights a critical implementation rule for the Lore exporter: the serialization logic must aggressively prune and strip empty or null tags (e.g., preventing the export of self-closing tags like \<subjects/\> or empty nodes like \<keywords\>\</keywords\>) to maintain backward compatibility with legacy viewer installations.

### **1.3 Timecode Format and Precision Expectations**

The treatment of temporal data within the XML schema is the central mechanism of its synchronizing function. As defined by the schema's underlying structural typologies and practical implementation paradigms, time units must be carefully formatted depending on the specific block being serialized.10  
Within the \<index\> block, which maps thematic topics to specific moments in the media, timecodes inside the \<point\>\<time\> elements are conventionally formatted as standard HH:MM:SS strings (e.g., 00:18:30).17 Because indexing is typically a macro-level navigational aid, millisecond precision is unnecessary and generally discarded in favor of whole-second integers.  
Within the legacy \<transcript\> block, the \<sync time="HH:MM:SS"\> tags similarly relied on integer precision.13 Because human transcribers mapped these timestamps manually during the legacy era, sub-second precision was impossible to achieve reliably.  
However, the adoption of the \<vtt\_transcript\> tag in OHMS XML 6.0 leverages the WebVTT standard format, requiring the structure HH:MM:SS.mmm (representing hours, minutes, seconds, and milliseconds to exactly three decimal places).10 This offers millisecond precision, aligning perfectly with the float outputs generated by the faster-whisper transcription engine. When Lore builds the XML payload, the float variables representing start and end times must be cast, formatted, and zero-padded to match the rigorous syntax of the WebVTT specification before being wrapped in CDATA tags within the XML tree.

### **1.4 Character Encoding and Syntax Requirements**

The mandated character encoding for OHMS XML files is UTF-8.11 Failure to strictly adhere to UTF-8 encoding and XML entity escaping protocols is a known, primary vector for ingestion failure within archival workflows. Historically, archivists have triggered severe parsing bugs and database exceptions when attempting to re-import OHMS XML files containing unescaped characters.9  
An update specifically deployed in OHMS v2.1.39 addressed catastrophic anomalies caused by unescaped ampersands (&) residing in the XML payload.9 A robust Python serializer must sanitize the transcription text and indexing output by escaping HTML and XML entities prior to constructing the ElementTree. The required entity conversions include replacing & with &, \< with \<, \> with \>, " with ", and ' with '. Furthermore, line-break filtering must be accounted for within the \<partial\_transcript\> tags of the index block to avoid breaking the presentation layer logic in the OHMS Viewer.9

### **1.5 Speaker Diarization and Labeling Mechanisms**

The OHMS XML schema does not possess a dedicated, proprietary XML tag for speaker attribution independent of the transcript text block. The schema relies on the formatting of the text itself to convey speaker identity.  
Under the legacy XML 5.4 model, speaker identities were simply embedded directly within the text payload as structural formatting, often utilizing all-caps identifiers followed by a colon (e.g., JOHN DOE: Well, the town was different then.).1  
Under the XML 6.0 model incorporating the \<vtt\_transcript\> element, speaker labeling relies entirely on the WebVTT standard tag \<v \> syntax. For example, a tagged line would appear as: \<v Dr. Smith\> How are you today?.10 Therefore, if Lore extracts speaker diarization data alongside its faster-whisper transcription, it must map this data directly into the VTT payload string syntax, rather than attempting to find a native OHMS XML element to house the speaker metadata.

### **1.6 Annotated OHMS XML Reference Architecture**

An operational OHMS XML file is divided into primary functional blocks: the root envelope, the \<record\> metadata envelope, the \<index\> block for topical segments, and the \<transcript\> (or \<vtt\_transcript\>) block for the verbatim text.2 To provide exact structural guidance for the Lore application's export module, an exhaustive, annotated synthesis of a modern OHMS XML file is provided below. This structure represents the optimal output for an edge-computed transcription system targeting modern repository compatibility.

XML  
\<?xml version="1.0" encoding="UTF-8"?\>  
\<ROOT xmlns\="https://www.weareavp.com/nunncenter/ohms"   
      xmlns:xsi\="http://www.w3.org/2001/XMLSchema-instance"   
      xsi:schemaLocation\="https://www.weareavp.com/nunncenter/ohms/ohms.xsd"\>  
      
    \<record\>  
        \<id\>00107334\</id\>  
        \<dt\>2025-10-15\</dt\>  
        \<accession\>OHP-0020\</accession\>  
        \<cms\_record\_id\>OHP-0020\</cms\_record\_id\>  
        \<collection\_id\>Latah-County-01\</collection\_id\>  
        \<collection\_link\>https://repository.institution.edu/latah-county\</collection\_link\>  
          
        \<title\>Interview with Severa Miyars Jacobs\</title\>  
        \<interviewer\>Dr. Elena Rostova\</interviewer\>  
        \<interviewee\>Severa Miyars Jacobs\</interviewee\>  
        \<format\>audio/mp3\</format\>  
          
        \<media\_host\>Aviary\</media\_host\>  
        \<media\_url\>https://institution.aviaryplatform.com/media/00107334.mp3\</media\_url\>  
          
        \<repository\>Louie B. Nunn Center for Oral History\</repository\>  
        \<rights\>All rights to the interviews, including legal title, copyrights and literary property rights, have been transferred to the University of Kentucky Libraries.\</rights\>  
        \<usage\>Interviews may only be reproduced with permission.\</usage\>  
        \<acknowledgement\>Funded by the KOHC Preservation Grant.\</acknowledgement\>  
        \<user\_note\>Audio quality degrades at 00:45:00 due to source tape degradation.\</user\_note\>

        \<index\>  
            \<point\>  
                \<time\>00:05:33\</time\>  
                \<title\>Recycling and Household Economies\</title\>  
                \<partial\_transcript\>What are some examples you remember of using recycled materials around the home...\</partial\_transcript\>  
                \<synopsis\>Jacobs talks about recycling materials during the depression, such as clothes and aluminum pans.\</synopsis\>  
                \<keywords\>Aluminum cans; Aluminum pans; Quilts; Resourcefulness\</keywords\>  
                \<subjects\>Environmentalism--History.; Recycling (Waste, etc.)\</subjects\>  
                \<gps\>36.0726, \-79.7920\</gps\>  
            \</point\>  
        \</index\>

        \<vtt\_transcript\>  
            \<\!\]\>  
        \</vtt\_transcript\>  
    \</record\>  
\</ROOT\>

## **2\. The OHMS Ecosystem: Platforms and Repositories**

Because the core OHMS application functions purely as a metadata preparation and synchronization environment—not as a permanent digital preservation repository—the generated XML files must be designed to interface with a host institution's Content Management System (CMS).7 Lore's output must navigate this fragmented ecosystem, ensuring that the generated XML is compatible with a wide array of ingestion methodologies.

### **2.1 Ecosystem Integration Matrix**

The following table synthesizes the varying levels of native and customized support across the dominant institutional repository platforms.

| Repository Platform | Support Status | Integration Methodology | Technical Implementation Notes |
| :---- | :---- | :---- | :---- |
| **Omeka S** | Native via Plugin | OhmsEmbed module (developed by the Omeka team).20 | Requires uploading the OHMS XML file as the primary media attachment on an item.22 Utilizes the ohms.js client-side viewer.23 Extracts text for global search via JSON Pointer mapping.22 |
| **Omeka Classic** | Native via Plugin | OHMSObject, OHMSExportPlugin, Philly themes.5 | Legacy support mechanism. Relies heavily on HTML iframes embedded directly into public-facing PHP themes.25 |
| **CONTENTdm** | Advanced Customization | JavaScript Cookbook Recipes via Website Configuration Tool.6 | Ingests the XML as a generic object and utilizes a custom JavaScript event hook to render the OHMS viewer in an iframe, allowing oral histories to reside alongside standard digital collections.6 |
| **Aviary (AVP)** | Fully Native | Application integrated directly into the cloud platform architecture.8 | Acquired OHMS in September 2023\. Supports OHMS XML 6.0 natively, reading \<vtt\_transcript\> tags seamlessly.10 |
| **AtoM (Access to Memory)** | Implicit Support | CSV Batch Ingest \+ Static XML Web Hosting.7 | Institutions host the XML file statically on an independent server and link it via a URL wrapper to a standalone installation of the OHMS Viewer.2 |

### **2.2 Omeka S Integration Dynamics**

The most prominent and accessible open-source integration exists within the Omeka S ecosystem. Recognizing the widespread adoption of OHMS, the Omeka core development team took full ownership of the OHMS plugin suite, releasing the refined OhmsEmbed module into the production environment in late 2024 to early 2025\.28  
The integration architecture for Omeka S represents a highly streamlined workflow. To deposit an oral history, the archivist must first configure the Omeka S installation to accept XML as an allowed media file extension.22 Subsequently, the archivist uploads the Lore-generated OHMS XML file directly to an Omeka item, designating it as the primary media asset.22  
Rather than relying on clunky server-side PHP processing, the OhmsEmbed module utilizes ohms.js, a specialized client-side JavaScript viewer, to parse the XML dynamically within the user's browser.23 Furthermore, the ingestion pipeline utilizes JSON Pointer syntax within the module's mapping configuration. This allows Omeka's native Solr or database-driven global search engines to extract and index the textual payloads contained within the XML's transcript and indexing nodes, making the oral history fully text-searchable alongside standard archival documents.22

### **2.3 CONTENTdm Integration Dynamics**

For institutions utilizing OCLC's widely adopted CONTENTdm platform, OHMS integration bypasses traditional server-side plugin architecture entirely, relying instead on advanced client-side rendering manipulations.6  
This integration is formalized through the CONTENTdm "Customization Cookbook".26 System administrators must upload a custom JavaScript payload via the CONTENTdm Website Configuration Tool.6 During the lifecycle event of a webpage loading, this custom script detects the presence of an OHMS-designated item, intercepts the standard CONTENTdm rendering pathway, and dynamically injects the OHMS Viewer iframe into the Document Object Model (DOM).6 This approach means that Lore’s OHMS XML items can be placed into existing CONTENTdm collections that contain other kinds of URL items or standard photographic records without disrupting the repository’s overarching structure.6

### **2.4 Aviary Integration and the Transition to Native Platforms**

In September 2023, the OHMS application was formally integrated into the Aviary platform, a cloud-based audiovisual access system developed by AVP.8 Aviary now acts as the primary host for the OHMS web application, providing native support for the new OHMS XML 6.0 standard and its WebVTT transcript format.10 While Aviary offers its own bulk import formats and CSV data structures for organizations lacking standard formats 3, the platform retains absolute backward and forward compatibility with OHMS XML.10 For Lore, this means that outputting standard OHMS XML guarantees compatibility with the most advanced, commercially supported oral history platform available today.

### **2.5 Deprecation Risks and Future-Proofing**

Within the academic oral history sector, there is virtually no immediate deprecation risk for the OHMS XML format itself. The format is heavily institutionalized, supporting hundreds of thousands of hours of digitized material across major universities and historical societies.1  
However, there is a severe deprecation risk regarding the *methods of timestamping* utilized within the XML. The deprecation of the manual "low beep/high beep" one-minute sync paradigm in favor of the WebVTT transcription standard (which ushered in XML 6.0) indicates a clear evolutionary path.10 Tools that continue to generate legacy \<sync\> tags will eventually face degraded support, rendering anomalies, or forced conversion algorithms in modern viewing interfaces.10 By engineering Lore to generate output compliant with OHMS XML 6.0—specifically leveraging the \<vtt\_transcript\> element—developers ensure the application avoids technical debt and remains fully aligned with the ecosystem's future trajectory.

## **3\. Competing and Complementary Standards**

While OHMS XML remains the dominant, purpose-built standard for complex oral history presentation, other timed-text formats exist within the digital media landscape. Understanding the limitations of these alternative formats is critical for justifying Lore’s reliance on OHMS XML as a primary export target.

### **3.1 OHMS XML vs. WebVTT vs. SRT**

| Technical Feature | OHMS XML (Version 6.0) | WebVTT (.vtt) | SubRip Subtitle (.srt) |
| :---- | :---- | :---- | :---- |
| **Primary Architectural Use Case** | Interactive oral history archiving, multifaceted semantic search, and structural conceptual indexing.2 | Native HTML5 \<track\> element display for standard browser-based video playback.4 | Basic television and film subtitling, ensuring universal local video player compatibility. |
| **Timecode Precision** | Granular (Milliseconds via nested VTT) and Legacy (Whole Seconds).10 | Milliseconds (HH:MM:SS.mmm). | Milliseconds (HH:MM:SS,mmm). Note the comma delimiter. |
| **Speaker Diarization Support** | Native support via nested VTT tag integration.10 | Native standard support (\<v Speaker Name\> Text). | Unofficial convention only; relies on plaintext formatting (e.g., Speaker:). |
| **Topical/Semantic Indexing** | Native hierarchical mapping of concepts to specific timestamps via the \<index\> block.19 | Lacks semantic indexing capabilities entirely. Purely presentational. | Lacks semantic indexing capabilities entirely. Purely presentational. |
| **Institutional Deposit Acceptance** | Mandated and supported by major archival frameworks (Omeka S, CONTENTdm, Aviary).6 | Increasingly accepted for basic caption generation workflows 10, but lacks descriptive metadata. | Rarely accepted as a primary preservation format due to a total lack of metadata extensibility. |

The fundamental, disqualifying limitation of both WebVTT and SRT formats is their inability to carry contextual, administrative, and descriptive metadata. WebVTT is strictly a presentation layer technology designed to render timed text over HTML5 media.4 It lacks the extensible capability to encapsulate Dublin Core metadata, biographical subject details, segment synopses, Library of Congress Subject Headings, or spatial GPS coordinates.11 OHMS XML solves this by providing a comprehensive XML wrapper that houses the WebVTT presentation logic alongside the deep semantic logic required by archivists.1

### **3.2 JSON-LD, Schema.org, and Dublin Core Sidecars**

While OHMS XML contains an internal \<record\> block for object-level metadata (such as titles, interviewers, and accessions) 12, modern institutional ingestion architectures often rely on independent metadata sidecars for integration into broader global cataloging networks. This is highly relevant to the Lore application’s integration with local collections management systems like Cache & Carry.  
Archival implementations frequently use a customized element set based heavily on the Dublin Core standard to map oral history data into the broader CMS infrastructure.15 When OHMS files are imported into Omeka S, the XML file is technically treated as a media attachment, while the core object record relies on standard semantic ontologies to define the item.22  
Furthermore, while there are emerging JSON-LD and schema.org profiles aimed at standardizing oral history metadata across the semantic web, OHMS XML remains the required format for the *viewer interface*. Therefore, it is highly recommended that Lore serializes a Dublin Core XML or JSON-LD sidecar alongside the OHMS XML payload. The OHMS XML file satisfies the requirements of the visual rendering engine (the OHMS Viewer), while the JSON-LD sidecar guarantees interoperability with standard REST APIs, linked open data networks, and object lifecycle management tools like Cache & Carry. This dual-export strategy ensures comprehensive metadata portability.

## **4\. OHMS Index vs. Transcript: Operational Use Cases**

A critical conceptual and technical distinction within the OHMS architecture is the separation of the Transcript and the Index.14 Understanding how institutions deploy these elements dictates how Lore should structure its automated output.

### **4.1 Epistemological and Functional Differences**

The **Transcript Block** contains the verbatim, word-for-word textual record of the interview, strictly synchronized to the media via timecodes. Its primary function is exact recall, accessibility compliance (e.g., ADA standards for the deaf and hard of hearing), and full-text searchability.19  
The **Index Block**, conversely, functions as an interactive finding aid and an epistemological map of the interview. It divides the continuous media into digestible, topic-based segments (often referred to as "stories" or "chapters").1 Each segment receives a descriptive title, a localized synopsis, keywords, and controlled vocabulary subject headings (such as those from the Library of Congress or the AFS Ethnographic Thesaurus).9 The index essentially maps natural language conversation to descriptive and meaningful concepts, allowing a researcher to bypass hours of audio and jump immediately to a discussion on a specific historical event.19

### **4.2 Application Realities and Labor Costs**

A valid OHMS XML file is not strictly required to contain both elements to pass validation.7  
Creating a high-quality semantic index requires immense human labor, specialized subject matter expertise, and deep engagement with controlled vocabularies. Because of this high cost, many institutions generate **Transcript-Only** OHMS XML files.13 If a Lore output file contains only the \<record\> wrapper and the \<vtt\_transcript\> block, it is entirely valid and will render correctly. The OHMS Viewer will automatically detect the absence of the \<index\> block and remove the Index toggle tab from the user interface.9  
Conversely, some institutions opt for **Index-Only** processing, allowing users to browse topics without bearing the financial or computational cost of generating a verbatim transcript.19 Utilizing both elements simultaneously offers the ultimate research experience, allowing the user to toggle between the broad thematic strokes of the index and the exact wording of the transcript.22  
For the Lore application, because the faster-whisper machine learning model fundamentally outputs verbatim text and accurate timestamps, exporting a Transcript-Only OHMS XML file serves as the most direct, schema-compliant baseline. Generating an \<index\> block programmatically would necessitate bridging the transcription data with a secondary Large Language Model (LLM) pipeline to extract thematic topics, summarize conversational segments into a \<synopsis\>, and deduce applicable \<keywords\>. While technically feasible, omitting the index entirely is standard archival practice and fully supported by the schema.

## **5\. Oral History Association: Ethical and Metadata Standards**

Producing technically compliant XML is insufficient if the output violates the ethical and legal frameworks governing oral history administration. Lore’s export architecture must support the ingestion of ethical metadata.

### **5.1 Rights, Licensing, and Usage Statements**

Oral histories are unique archival assets because they involve living human subjects, requiring explicit, documented consent and the legal transfer of title.18 The metadata payload within the OHMS XML \<record\> must accurately reflect the disposition of these rights. The Oral History Association (OHA) and associated repository best practices emphasize the inclusion of standardized rights statements directly linked to the digital object, ensuring that researchers understand the legal boundaries of the material.33  
Within the OHMS XML schema, this critical information is handled via the \<rights\> and \<usage\> elements.

* **Rights Statement Example:** "All rights to the interviews, including but not restricted to legal title, copyrights and literary property rights, have been transferred to the University of Kentucky Libraries." 18  
* **Usage Statement Example:** "Interviews may only be reproduced with permission from Louie B. Nunn Center for Oral History." 18

While Creative Commons variants (e.g., CC BY-NC) are highly encouraged for public-facing digital humanities projects 32, many interviews contain sensitive, defamatory, or restricted contents that necessitate stricter usage statements or "closed" archival status.33  
Regarding speaker consent documentation, linking physical or digital consent forms directly inside the transcript record is not formally required by the OHMS XML schema. Best practices dictate that the \<collection\_link\> or \<cms\_record\_id\> correlates to a master CMS record (such as an entry in Cache & Carry) where the actual consent documentation is privately secured and access-restricted.12

### **5.2 The Disambiguation of "TRAILS"**

Regarding the inquiry into "TRAILS" (Transfer of Rights and Acquisition in Library Systems), a rigorous analysis of the contemporary archival landscape and available oral history project documentation reveals that this acronym does not refer to an established library metadata rights standard. In the context of oral history projects across municipal and state archives, "TRAILS" predominantly refers to physical recreational paths, such as the "Rails to Trails" initiatives documented extensively in Greene County, Ohio, and Wyoming oral history archives.34 Rights and acquisitions in oral history are governed instead by standardized NLA Rights Agreements 33 and conventional institutional Deed of Gift forms. Therefore, Lore's development team does not need to parse, implement, or comply with a "TRAILS" metadata standard for rights management.

## **6\. Practical Implementation Guidance for Application Developers**

Translating raw transcription arrays and float variables from the faster-whisper engine into a compliant, robust OHMS XML document involves navigating several programmatic hurdles.

### **6.1 Evaluating Implementation Complexity**

The proposition in the project scope that an OHMS serialiser requires merely "100–150 lines of straightforward XML generation" is highly optimistic if targeting robust, enterprise-grade compliance.  
While creating a barebones XML string via an lxml or xml.etree.ElementTree library in Python can technically be accomplished in under 100 lines, handling the inherent edge cases significantly inflates architectural complexity. A production-ready Python exporter module (lore/src/exporters/ohms.py) must account for the following programmatic operations:

1. **VTT Conversion and Time Casting:** Transforming the continuous \[start, end\] float segment tuples outputted by faster-whisper into the precise, zero-padded HH:MM:SS.mmm string formatting required for the OHMS XML 6.0 \<vtt\_transcript\> tag.10 This requires custom datetime parsing functions.  
2. **Entity Sanitization Pipeline:** Recursively scrubbing transcription text for unescaped characters (specifically &, \<, and \>) to prevent fatal parsing errors upon CMS ingestion.9  
3. **Empty Tag Pruning Logic:** Implementing recursive traversal of the XML tree to strip null, empty, or whitespace-only metadata fields (like an empty \<user\_note\> or \<gps\>) prior to serialization. This prevents UI skewing in legacy OHMS Viewers.9  
4. **CDATA Encapsulation:** The WebVTT text block must be wrapped in XML \<\!\]\> tags. Standard Python XML libraries often struggle with native CDATA generation, requiring custom serialization overrides or the use of specific lxml extensions.

### **6.2 Common Submission Errors and Implementation "Gotchas"**

Based on archivist reports, issue trackers, and platform patch logs, the most common errors encountered during OHMS XML ingest into Omeka S or CONTENTdm include:

1. **Timecode Desynchronization Post-Processing:** The timestamps embedded in the OHMS XML map directly to specific temporal locations in the streaming media file. If an archivist utilizes Lore to generate the XML transcript, and subsequently uses an audio editor to trim silence, normalize volume, or apply noise reduction to the source audio, the overall duration of the media changes. This irreparably disrupts the correspondence between the XML timecode markers and the media content, causing the OHMS viewer to jump to incorrect moments.7  
2. **Unescaped Ampersands in Text Nodes:** As noted previously, the presence of an unescaped & instead of & within \<partial\_transcript\> or \<title\> blocks causes immediate, unrecoverable validation failures upon XML import.9  
3. **Media Protocol Mismatch:** The URL supplied in the \<media\_url\> field historically required specific protocols. For example, legacy bugs rejected HTTPS prefixes for YouTube media, causing player initialization failure.11 Lore must ensure that the generated URLs are well-formed and compliant with modern CORS (Cross-Origin Resource Sharing) policies dictated by the host institution.  
4. **Line Break Formatting:** Feeding unbroken, massive text blocks into partial transcripts without line-break filtering can break the OHMS Viewer layout. Formatting must be preserved or safely stripped.9

### **6.3 Open-Source Reference Generators**

Several open-source projects have successfully attempted to parse or generate OHMS XML data. These repositories can serve as architectural references for constructing Lore's ohms.py module:

* **The Transcript Indexing Module (TIM):** Designed specifically to generate OHMS XML indexing metadata independently of the official OHMS web application. It functions as a stand-alone tool for segmenting and basic XML mark-up, proving that the offline generation of OHMS XML is practically viable.38  
* **convertToOHMS.rb (mrascher):** A lightweight Ruby script available on GitHub designed to convert a coded transcript text file directly into OHMS XML.39 It provides a solid baseline for regex-based timecode mapping and string manipulation.  
* **OHMSExportPlugin (Omeka Classic):** An open-source plugin developed by Eric C. Weig that extracts oral history metadata from the Omeka database and serializes it into OHMS XML.24 Reviewing the PHP source code provides a definitive guide on how empty fields and CDATA wrappers should be handled programmatically to ensure CMS acceptance.  
* **ohms.js (Omeka S):** The client-side JavaScript viewer currently maintained by the Omeka development team.23 Analyzing the parsing logic in ohms.js provides the ultimate source of truth regarding exactly which XML nodes are strictly required for the player to initialize without throwing a fatal exception.

## **Conclusion**

The integration of the edge-computed Lore application into the institutional oral history ecosystem hinges entirely on the semantic and syntactic integrity of its XML output. The OHMS XML standard, while relatively straightforward in its structural schema, operates as the critical connective tissue between raw, unstructured media and highly navigable historical metadata.  
To ensure maximum interoperability with dominant repository platforms like Omeka S, CONTENTdm, and Aviary, the Lore application must implement an exporter module that defaults to the modern OHMS XML 6.0 standard. By encapsulating faster-whisper outputs within the \<vtt\_transcript\> element for precise millisecond synchronization, strictly adhering to UTF-8 escaping protocols, pruning empty metadata nodes prior to serialization, and supplementing the primary XML payload with an independent JSON-LD or Dublin Core sidecar, the application will bypass common archival ingestion errors. This rigorous approach guarantees the delivery of an unassailable, preservation-ready digital asset that meets the highest standards of the global oral history community.

#### **Works cited**

1. Synchronizing Oral History Text and Speech: A Tools Overview, accessed June 4, 2026, [https://academicworks.cuny.edu/cgi/viewcontent.cgi?article=1043\&context=jj\_pubs](https://academicworks.cuny.edu/cgi/viewcontent.cgi?article=1043&context=jj_pubs)  
2. OHMS Oral History Metadata Synchronizer, accessed June 4, 2026, [https://www.oralhistoryonline.org/](https://www.oralhistoryonline.org/)  
3. Aviary: An Access Platform For Audiovisual Content \- iPRES 2019, accessed June 4, 2026, [https://ipres2019.org/static/pdf/iPres2019\_paper\_56.pdf](https://ipres2019.org/static/pdf/iPres2019_paper_56.pdf)  
4. From Preservation to access in one steP Aviary: An Access Platform For Audiovisual Content \- PHAIDRA, accessed June 4, 2026, [https://services.phaidra.univie.ac.at/api/object/o:1079790/download](https://services.phaidra.univie.ac.at/api/object/o:1079790/download)  
5. Using OHMS with Omeka \- Oral History Metadata Synchronizer, accessed June 4, 2026, [https://www.oralhistoryonline.org/documentation/omeka/](https://www.oralhistoryonline.org/documentation/omeka/)  
6. Embed OHMS viewer \- CONTENTdm \- OCLC Support, accessed June 4, 2026, [https://help.oclc.org/Metadata\_Services/CONTENTdm/Advanced\_website\_customization/Customization\_cookbook/Embed\_OHMS\_viewer](https://help.oclc.org/Metadata_Services/CONTENTdm/Advanced_website_customization/Customization_cookbook/Embed_OHMS_viewer)  
7. OHMS (Oral History Metadata Synchronizer) User Guide, accessed June 4, 2026, [https://www.oralhistoryonline.org/wp-content/uploads/2020/11/@OHMS\_user\_guide\_master\_v3-8-3.pdf](https://www.oralhistoryonline.org/wp-content/uploads/2020/11/@OHMS_user_guide_master_v3-8-3.pdf)  
8. Resources and Guides | OHMS: Oral History Metadata Synchronizer, accessed June 4, 2026, [https://www.oralhistoryonline.org/documentation/](https://www.oralhistoryonline.org/documentation/)  
9. Releases and Versions | OHMS \- Oral History Metadata Synchronizer, accessed June 4, 2026, [https://www.oralhistoryonline.org/help/release/](https://www.oralhistoryonline.org/help/release/)  
10. OHMS, WebVTT, and the Transcript Editor of my Dreams \- Digital Omnium, accessed June 4, 2026, [https://digitalomnium.com/ohms-vtt-and-the-transcript-editor-of-my-dreams/](https://digitalomnium.com/ohms-vtt-and-the-transcript-editor-of-my-dreams/)  
11. OHMS Application Update v.2.2.20 \- Oral History Metadata Synchronizer, accessed June 4, 2026, [https://www.oralhistoryonline.org/ohms-application-update-v-2-2-20/](https://www.oralhistoryonline.org/ohms-application-update-v-2-2-20/)  
12. Performing loop on js objects and then use map method instead of calling component again and again \- Stack Overflow, accessed June 4, 2026, [https://stackoverflow.com/questions/69964412/performing-loop-on-js-objects-and-then-use-map-method-instead-of-calling-compone](https://stackoverflow.com/questions/69964412/performing-loop-on-js-objects-and-then-use-map-method-instead-of-calling-compone)  
13. The Oral History Metadata Synchronizer (OHMS): Enhancing Discoverability for Oral History Collections \- Stephanie Sapienza, accessed June 4, 2026, [http://stephaniesapienza.com/documents/ohms-workshop.pdf](http://stephaniesapienza.com/documents/ohms-workshop.pdf)  
14. OHMS Tutorial | Doing Digital History: 2016, accessed June 4, 2026, [https://history2016.doingdh.org/ohms-tutorial/](https://history2016.doingdh.org/ohms-tutorial/)  
15. Building Digital Oral History Collections, accessed June 4, 2026, [https://www.amicalnet.org/assets/files/amical-2019-kahale.pdf](https://www.amicalnet.org/assets/files/amical-2019-kahale.pdf)  
16. https://images-na.ssl-images-amazon.com/images/G/01/rainier/help/xsd/release\_1\_9/amzn-base.\_TTH\_.xsd, accessed June 4, 2026, [https://images-na.ssl-images-amazon.com/images/G/01/rainier/help/xsd/release\_1\_9/amzn-base.\_TTH\_.xsd](https://images-na.ssl-images-amazon.com/images/G/01/rainier/help/xsd/release_1_9/amzn-base._TTH_.xsd)  
17. Interview with Wilson Francis, July 17, 1989 \- Louie B. Nunn Center for Oral History, accessed June 4, 2026, [https://nunncenter.net/ohms-spokedb/render.php?cachefile=1989oh161\_env015\_ohm.xml](https://nunncenter.net/ohms-spokedb/render.php?cachefile=1989oh161_env015_ohm.xml)  
18. Interview with William Brooks, \- Louie B. Nunn Center for Oral History, accessed June 4, 2026, [https://nunncenter.net/ohms-spokedb/render.php?cachefile=2008oh026\_pride071\_ohm.xml](https://nunncenter.net/ohms-spokedb/render.php?cachefile=2008oh026_pride071_ohm.xml)  
19. Search, Explore, Connect: Using OHMS to Enhance Access to Oral History \- UKnowledge, accessed June 4, 2026, [https://uknowledge.uky.edu/cgi/viewcontent.cgi?article=1293\&context=libraries\_facpub](https://uknowledge.uky.edu/cgi/viewcontent.cgi?article=1293&context=libraries_facpub)  
20. OHMS Embed \- Omeka S, accessed June 4, 2026, [https://omeka.org/s/modules/OhmsEmbed/](https://omeka.org/s/modules/OhmsEmbed/)  
21. omeka-s-modules repositories \- GitHub, accessed June 4, 2026, [https://github.com/orgs/omeka-s-modules/repositories?type=all](https://github.com/orgs/omeka-s-modules/repositories?type=all)  
22. OHMS Embed \- Omeka S User Manual, accessed June 4, 2026, [https://omeka.org/s/docs/user-manual/modules/ohmsembed/](https://omeka.org/s/docs/user-manual/modules/ohmsembed/)  
23. Pull requests · omeka/ohms.js · GitHub, accessed June 4, 2026, [https://github.com/omeka/ohms.js/pulls](https://github.com/omeka/ohms.js/pulls)  
24. UpgradeToOmekaS/\_data/omeka\_plugins.csv at master \- GitHub, accessed June 4, 2026, [https://github.com/Daniel-KM/UpgradeToOmekaS/blob/master/\_data/omeka\_plugins.csv](https://github.com/Daniel-KM/UpgradeToOmekaS/blob/master/_data/omeka_plugins.csv)  
25. libmanuk/OHMSObject: An Omeka plugin that embeds an OHMS Object via a URL in the ... \- GitHub, accessed June 4, 2026, [https://github.com/libmanuk/OHMSObject](https://github.com/libmanuk/OHMSObject)  
26. Advanced website customization \- CONTENTdm \- OCLC Support, accessed June 4, 2026, [https://help.oclc.org/Metadata\_Services/CONTENTdm/Advanced\_website\_customization](https://help.oclc.org/Metadata_Services/CONTENTdm/Advanced_website_customization)  
27. gsu-library/contentdm-hosted-files: CSS, JavaScript, images, and template files for ... \- GitHub, accessed June 4, 2026, [https://github.com/gsu-library/contentdm-hosted-files](https://github.com/gsu-library/contentdm-hosted-files)  
28. New OHMS Modules/Plugins \- Development \- Omeka Forum, accessed June 4, 2026, [https://forum.omeka.org/t/new-ohms-modules-plugins/23971](https://forum.omeka.org/t/new-ohms-modules-plugins/23971)  
29. Connecting Historical and Digital Frontiers: Enhancing Access to the Latah County Oral History Collection Utilizing OHMS (Oral History Metadata Synchronizer) and Isotope \- The Code4Lib Journal, accessed June 4, 2026, [https://journal.code4lib.org/articles/10643](https://journal.code4lib.org/articles/10643)  
30. Jaycie N. Vos. The Development of a Shared Metadata Standard for Use in Oral History, accessed June 4, 2026, [https://cdr.lib.unc.edu/downloads/dr26z198c](https://cdr.lib.unc.edu/downloads/dr26z198c)  
31. Case Study 9.3: Oral History Metadata Synchronizer—A Transcript Solution, accessed June 4, 2026, [https://opentext.wsu.edu/accessibility-case-studies/chapter/case-study9-ohms/](https://opentext.wsu.edu/accessibility-case-studies/chapter/case-study9-ohms/)  
32. OHMS Embed \- Omeka Classic User Manual, accessed June 4, 2026, [https://omeka.org/classic/docs/Plugins/OhmsEmbed/](https://omeka.org/classic/docs/Plugins/OhmsEmbed/)  
33. dh2015/xml/THOMSON\_ALISTAIR\_Making\_Digital\_Aural\_History.xml at master \- GitHub, accessed June 4, 2026, [https://github.com/ADHO/dh2015/blob/master/xml/THOMSON\_ALISTAIR\_Making\_Digital\_Aural\_History.xml](https://github.com/ADHO/dh2015/blob/master/xml/THOMSON_ALISTAIR_Making_Digital_Aural_History.xml)  
34. Oral History Project \- New River Gorge National Park & Preserve (U.S. National Park Service), accessed June 4, 2026, [https://www.nps.gov/neri/learn/historyculture/oral-history-project.htm](https://www.nps.gov/neri/learn/historyculture/oral-history-project.htm)  
35. Oral Histories | Greene County, OH \- Official Website, accessed June 4, 2026, [https://www.greenecountyohio.gov/1546/Oral-Histories](https://www.greenecountyohio.gov/1546/Oral-Histories)  
36. Oral History Collection \- Wyoming State Archives, accessed June 4, 2026, [https://wyoarchives.wyo.gov/index.php/find-it-in-the-archives/oral-history-collection](https://wyoarchives.wyo.gov/index.php/find-it-in-the-archives/oral-history-collection)  
37. OHMS (Oral History Metadata Synchronizer) User Guide, accessed June 4, 2026, [https://www.oralhistoryonline.org/wp-content/uploads/2023/09/OHMS\_Aviary\_user\_guide\_Master\_Aviary\_v2.0\_09\_17.pdf](https://www.oralhistoryonline.org/wp-content/uploads/2023/09/OHMS_Aviary_user_guide_Master_Aviary_v2.0_09_17.pdf)  
38. Oral History Indexing \- Personal Websites \- University at Buffalo, accessed June 4, 2026, [https://www.acsu.buffalo.edu/\~bert/downloads/Oral\_History\_Indexing/OHI-Lambert-2023.pdf](https://www.acsu.buffalo.edu/~bert/downloads/Oral_History_Indexing/OHI-Lambert-2023.pdf)  
39. Simple ruby script to convert a coded transcript to OHMS xml · GitHub, accessed June 4, 2026, [https://gist.github.com/mrascher/46ebe05d7973e43db90bba8044a0a5bc](https://gist.github.com/mrascher/46ebe05d7973e43db90bba8044a0a5bc)