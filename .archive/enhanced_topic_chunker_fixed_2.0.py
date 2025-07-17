# enhanced_topic_chunker_fixed.py
# This is the complete and corrected file. You can copy and paste this entire script.

import re
import time
import streamlit as st
from typing import List, Dict, Callable
from dataclasses import dataclass

@dataclass
class TopicSegment:
    """Represents a topic segment in the deposition"""
    topic: str
    start_idx: int
    end_idx: int
    text: str
    confidence: float
    method: str = "unknown"

class EnhancedTopicBasedChunker:
    """Reliable topic-based chunking for deposition transcripts"""

    def __init__(self,
                 llm_provider: str,
                 llm_model: str,
                 target_chunk_size: int = 8000,
                 min_chunk_size: int = 2000,
                 max_chunk_size: int = 12000,
                 overlap_size: int = 400,
                 quality_threshold: float = 0.7,
                 call_llm_func: Callable = None):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.quality_threshold = quality_threshold
        self.call_llm = call_llm_func
        self.processing_stats = {
            'llm_calls': 0,
            'fallback_count': 0,
            'processing_time': 0,
            'chunks_created': 0
        }

    def chunk_transcript(self, transcript_text: str) -> List[str]:
        """Main method with enhanced reliability"""
        start_time = time.time()

        if not transcript_text or len(transcript_text.strip()) < 100:
            st.warning("Text too short for topic-based chunking")
            return self._simple_chunk(transcript_text)

        try:
            st.info("Analyzing transcript structure...")
            topic_segments = self._multi_method_topic_detection(transcript_text)

            quality_score = self._evaluate_segmentation_quality(topic_segments)
            st.info(f"Segmentation quality score: {quality_score:.2f}")

            if quality_score < self.quality_threshold:
                st.warning("Applying segmentation corrections...")
                topic_segments = self._correct_segments(topic_segments)

            st.info("Optimizing chunk sizes...")
            sized_segments = self._optimize_segment_sizes(topic_segments)

            final_chunks = self._create_final_chunks_enhanced(sized_segments)
            final_chunks = self._validate_final_chunks(final_chunks)

            self.processing_stats['processing_time'] = time.time() - start_time
            self.processing_stats['chunks_created'] = len(final_chunks)

            st.success(f"Created {len(final_chunks)} chunks in {self.processing_stats['processing_time']:.1f}s")
            return final_chunks

        except Exception as e:
            st.error(f"Topic chunking failed: {e}")
            self.processing_stats['fallback_count'] += 1
            return self._simple_chunk(transcript_text)

    def _multi_method_topic_detection(self, text: str) -> List[TopicSegment]:
        """Use multiple methods and combine results"""
        all_segments = []
        if self.call_llm:
            try:
                llm_segments = self._identify_topics_llm_optimized(text)
                if llm_segments:
                    all_segments.extend(llm_segments)
            except Exception as e:
                st.warning(f"LLM detection failed: {e}")

        try:
            pattern_segments = self._pattern_based_segmentation_enhanced(text)
            if pattern_segments:
                if not all_segments:
                    all_segments = pattern_segments
                else:
                    all_segments = self._merge_detection_results(all_segments, pattern_segments)
        except Exception as e:
            st.warning(f"Pattern detection failed: {e}")

        if not all_segments:
            st.warning("All detection methods failed, using uniform segments")
            return self._create_uniform_segments(text)

        return all_segments

    def _identify_topics_llm_optimized(self, text: str) -> List[TopicSegment]:
        """Optimized LLM-based topic identification"""
        if not self.call_llm:
            raise Exception("LLM function not available")

        sample_size = min(15000, len(text))
        samples = [text[:sample_size//3], text[len(text)//2 - sample_size//6:len(text)//2 + sample_size//6], text[-sample_size//3:]] if len(text) > sample_size * 3 else [text[:sample_size]]

        prompt = f"""
        Analyze this deposition transcript and identify topic transitions.
        Common deposition topics:
        - Personal Information (name, address, background)
        - Employment History
        - Medical History (pre-incident)
        - The Incident/Accident
        - Post-Incident Events
        - Medical Treatment
        - Current Condition
        For each topic transition, provide:
        1. Topic name
        2. A unique phrase/marker near the transition
        3. Confidence (0.0-1.0)
        Format: TOPIC: [name] | MARKER: [phrase] | CONFIDENCE: [score]
        TRANSCRIPT SAMPLES:
        {' [...] '.join(samples)}
        """

        self.processing_stats['llm_calls'] += 1
        response = self.call_llm(prompt, self.llm_provider, self.llm_model)

        if response.startswith("Error:"):
            raise Exception(f"LLM call failed: {response}")

        return self._parse_llm_response(response, text)

    def _parse_llm_response(self, response: str, full_text: str) -> List[TopicSegment]:
        segments, lines, markers = [], response.split('\n'), []
        for line in lines:
            if 'TOPIC:' in line and 'MARKER:' in line:
                try:
                    parts = line.split('|')
                    topic, marker, confidence = '', '', 0.8
                    for part in parts:
                        if 'TOPIC:' in part: topic = part.split('TOPIC:')[1].strip()
                        elif 'MARKER:' in part: marker = part.split('MARKER:')[1].strip()
                        elif 'CONFIDENCE:' in part:
                            try: confidence = float(part.split('CONFIDENCE:')[1].strip())
                            except: pass
                    if topic and marker: markers.append({'topic': topic, 'marker': marker, 'confidence': confidence})
                except: continue

        last_pos = 0
        for i, marker_info in enumerate(markers):
            marker_pos = full_text.find(marker_info['marker'], last_pos)
            if marker_pos == -1: marker_pos = self._fuzzy_find(full_text, marker_info['marker'], last_pos)
            if marker_pos != -1 and marker_pos > last_pos:
                if i > 0:
                    segments.append(TopicSegment(topic=markers[i-1]['topic'], start_idx=last_pos, end_idx=marker_pos, text=full_text[last_pos:marker_pos], confidence=markers[i-1]['confidence'], method='llm'))
                last_pos = marker_pos
        if last_pos < len(full_text) and markers:
            segments.append(TopicSegment(topic=markers[-1]['topic'], start_idx=last_pos, end_idx=len(full_text), text=full_text[last_pos:], confidence=markers[-1]['confidence'], method='llm'))
        return segments

    def _pattern_based_segmentation_enhanced(self, text: str) -> List[TopicSegment]:
        PATTERNS = {
            'Personal Information': [r'(?:please\s+)?state\s+your\s+(?:full\s+)?name', r'what\s+is\s+your\s+name', r'where\s+do\s+you\s+live', r'(?:current\s+)?address'],
            'Employment': [r'where\s+do\s+you\s+work', r'(?:current\s+)?employer', r'what\s+is\s+your\s+(?:job|occupation)', r'employment\s+history'],
            'Medical History': [r'medical\s+history', r'(?:any\s+)?(?:prior\s+)?surgeries', r'health\s+conditions', r'medications'],
            'Incident Details': [r'(?:day|date)\s+of\s+the\s+(?:accident|incident)', r'tell\s+us\s+what\s+happened', r'describe\s+the\s+(?:accident|incident)'],
            'Current Condition': [r'current\s+condition', r'how\s+are\s+you\s+feeling', r'(?:still\s+)?treating', r'pain\s+level']
        }
        found_positions = []
        for topic, patterns in PATTERNS.items():
            for pattern in patterns:
                try:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        found_positions.append({'topic': topic, 'position': match.start(), 'text': match.group(), 'confidence': 0.7})
                except re.error: continue
        found_positions.sort(key=lambda x: x['position'])

        segments, last_pos = [], 0
        for i, pos_info in enumerate(found_positions):
            if pos_info['position'] > last_pos + 100:
                prev_topic = found_positions[i-1]['topic'] if i > 0 else 'Introduction'
                segments.append(TopicSegment(topic=prev_topic, start_idx=last_pos, end_idx=pos_info['position'], text=text[last_pos:pos_info['position']], confidence=0.7, method='pattern'))
                last_pos = pos_info['position']
        if last_pos < len(text) - 100:
            final_topic = found_positions[-1]['topic'] if found_positions else 'Content'
            segments.append(TopicSegment(topic=final_topic, start_idx=last_pos, end_idx=len(text), text=text[last_pos:], confidence=0.6, method='pattern'))
        return segments

    def _create_uniform_segments(self, text: str) -> List[TopicSegment]:
        segments = []
        for i in range(0, len(text), self.target_chunk_size - self.overlap_size):
            end = min(i + self.target_chunk_size, len(text))
            segments.append(TopicSegment(topic=f"Section {len(segments) + 1}", start_idx=i, end_idx=end, text=text[i:end], confidence=0.5, method='uniform'))
        return segments

    def _merge_detection_results(self, primary: List[TopicSegment], secondary: List[TopicSegment]) -> List[TopicSegment]:
        return primary if primary else secondary

    def _evaluate_segmentation_quality(self, segments: List[TopicSegment]) -> float:
        if not segments: return 0.0
        scores, sizes = [], [len(s.text) for s in segments]
        if len(sizes) > 1:
            avg_size = sum(sizes) / len(sizes)
            size_variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)
            scores.append(max(0, 1 - (size_variance ** 0.5) / avg_size))
        scores.append(sum(s.confidence for s in segments) / len(segments))
        total_length, expected_length = sum(len(s.text) for s in segments), segments[-1].end_idx - segments[0].start_idx if segments else 0
        scores.append(min(1.0, total_length / expected_length) if expected_length > 0 else 0)
        return sum(scores) / len(scores) if scores else 0.0

    def _correct_segments(self, segments: List[TopicSegment]) -> List[TopicSegment]:
        """Apply corrections to improve segmentation."""
        if not segments:
            return segments

        corrected = []
        # Create a mutable copy to work with
        mutable_segments = list(segments)

        for i, segment in enumerate(mutable_segments):
            # This check prevents the error. If the segment was already merged
            # and set to None, we just skip it.
            if segment is None:
                continue

            # Fix empty segments
            if not segment.text.strip():
                continue

            # Fix size issues by trying to merge with the next segment
            if len(segment.text) < self.min_chunk_size and i < len(mutable_segments) - 1:
                next_segment = mutable_segments[i + 1]
                # Also ensure the next segment isn't None
                if next_segment and (len(segment.text) + len(next_segment.text) < self.max_chunk_size):
                    merged = TopicSegment(
                        topic=f"{segment.topic} & {next_segment.topic}",
                        start_idx=segment.start_idx,
                        end_idx=next_segment.end_idx,
                        text=segment.text + "\n\n" + next_segment.text,
                        confidence=min(segment.confidence, next_segment.confidence),
                        method='corrected-merged'
                    )
                    corrected.append(merged)
                    mutable_segments[i + 1] = None  # Mark as merged
                    continue # Skip to the next iteration

            # If no merge happened, just add the original segment
            corrected.append(segment)

        # Filter out any None values that might remain
        return [s for s in corrected if s is not None]

    def _optimize_segment_sizes(self, segments: List[TopicSegment]) -> List[TopicSegment]:
        optimized = []
        for segment in segments:
            if len(segment.text) > self.max_chunk_size:
                optimized.extend(self._split_large_segment(segment))
            else:
                optimized.append(segment)

        final, i = [], 0
        while i < len(optimized):
            current = optimized[i]
            while (i < len(optimized) - 1 and len(current.text) < self.min_chunk_size and len(current.text) + len(optimized[i + 1].text) < self.max_chunk_size):
                next_seg = optimized[i + 1]
                current = TopicSegment(topic=f"{current.topic} + {next_seg.topic}", start_idx=current.start_idx, end_idx=next_seg.end_idx, text=current.text + "\n\n" + next_seg.text, confidence=min(current.confidence, next_seg.confidence), method='merged')
                i += 1
            final.append(current)
            i += 1
        return final

    def _split_large_segment(self, segment: TopicSegment) -> List[TopicSegment]:
        splits, paragraphs, current_split, split_count = [], segment.text.split('\n\n'), "", 0
        for para in paragraphs:
            if len(current_split) + len(para) < self.target_chunk_size:
                current_split += para + "\n\n"
            else:
                if current_split:
                    splits.append(TopicSegment(topic=f"{segment.topic} (Part {split_count + 1})", start_idx=segment.start_idx, end_idx=segment.start_idx + len(current_split), text=current_split.strip(), confidence=segment.confidence * 0.9, method='split'))
                    split_count += 1
                current_split = para + "\n\n"
        if current_split:
            splits.append(TopicSegment(topic=f"{segment.topic} (Part {split_count + 1})", start_idx=segment.start_idx, end_idx=segment.end_idx, text=current_split.strip(), confidence=segment.confidence * 0.9, method='split'))
        return splits if splits else [segment]

    def _create_final_chunks_enhanced(self, segments: List[TopicSegment]) -> List[str]:
        if not segments: return []
        final_chunks = []
        for i, segment in enumerate(segments):
            chunk_parts = [f"[TOPIC: {segment.topic}]", f"[Method: {segment.method}, Confidence: {segment.confidence:.2f}]"]
            if i > 0 and self.overlap_size > 0:
                prev_text = segments[i-1].text
                if prev_text:
                    overlap_text = prev_text[-min(self.overlap_size, len(prev_text)):]
                    sentences = re.split(r'(?<=[.!?])\s+', overlap_text)
                    if sentences and len(sentences[-1]) > 20: chunk_parts.append(f"\n[CONTEXT: ...{sentences[-1]}]\n")
            chunk_parts.append(segment.text)
            if i < len(segments) - 1 and self.overlap_size > 0:
                next_text = segments[i+1].text
                if next_text:
                    preview_text = next_text[:min(self.overlap_size, len(next_text))]
                    match = re.match(r'^.*?[.!?]', preview_text)
                    if match: chunk_parts.append(f"\n[CONTINUES: {match.group(0)}...]\n")
            final_chunks.append("\n".join(chunk_parts))
        return final_chunks

    def _validate_final_chunks(self, chunks: List[str]) -> List[str]:
        validated = []
        for chunk in chunks:
            if not chunk or len(chunk.strip()) < 50: continue
            if len(chunk) > self.max_chunk_size * 1.5:
                mid = len(chunk) // 2
                validated.extend([chunk[:mid], chunk[mid:]])
            else:
                validated.append(chunk)
        return validated if validated else chunks[:1]

    def _fuzzy_find(self, text: str, marker: str, start_pos: int = 0) -> int:
        if not marker: return -1
        words = marker.split()
        for num_words in range(len(words), 0, -1):
            pos = text.find(' '.join(words[:num_words]), start_pos)
            if pos != -1: return pos
        return -1

    def _simple_chunk(self, text: str) -> List[str]:
        if not text: return []
        chunks = [text[i:i + self.target_chunk_size] for i in range(0, len(text), self.target_chunk_size - self.overlap_size)]
        return [c for c in chunks if c] or [text]

    def get_stats(self) -> Dict:
        return self.processing_stats