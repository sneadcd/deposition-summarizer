# enhanced_topic_chunker_improved.py
# Improved version with performance optimizations, better error handling, and configurable parameters

import re
import time
import hashlib
import streamlit as st
from typing import List, Dict, Callable, Optional, Tuple
from dataclasses import dataclass, field
from functools import lru_cache

@dataclass
class TopicSegment:
    """Represents a topic segment in the deposition"""
    topic: str
    start_idx: int
    end_idx: int
    text: str
    confidence: float
    method: str = "unknown"
    
    def __post_init__(self):
        """Validate segment data"""
        if self.start_idx < 0 or self.end_idx < self.start_idx:
            raise ValueError(f"Invalid segment indices: {self.start_idx}-{self.end_idx}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

@dataclass
class ChunkerConfig:
    """Configuration for the topic chunker"""
    target_chunk_size: int = 8000
    min_chunk_size: int = 2000
    max_chunk_size: int = 12000
    overlap_size: int = 400
    quality_threshold: float = 0.7
    min_segment_gap: int = 100
    min_text_length: int = 100
    preview_sentence_min_length: int = 20
    emergency_split_threshold: float = 1.5
    
    def __post_init__(self):
        """Validate configuration"""
        if not (self.min_chunk_size <= self.target_chunk_size <= self.max_chunk_size):
            raise ValueError("Chunk sizes must satisfy: min <= target <= max")
        if self.overlap_size >= self.min_chunk_size:
            raise ValueError("Overlap size must be less than minimum chunk size")

class EnhancedTopicBasedChunker:
    """Improved topic-based chunking for deposition transcripts with caching and better error handling"""

    def __init__(self,
                 llm_provider: str,
                 llm_model: str,
                 config: Optional[ChunkerConfig] = None,
                 call_llm_func: Callable = None,
                 enable_caching: bool = True):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.config = config or ChunkerConfig()
        self.call_llm = call_llm_func
        self.enable_caching = enable_caching
        self._response_cache = {} if enable_caching else None
        self.processing_stats = {
            'llm_calls': 0,
            'cache_hits': 0,
            'fallback_count': 0,
            'processing_time': 0,
            'chunks_created': 0,
            'total_text_length': 0
        }
        
        # Compile regex patterns once for better performance
        self._compiled_patterns = self._compile_deposition_patterns()

    def _compile_deposition_patterns(self) -> Dict[str, List]:
        """Pre-compile regex patterns for better performance"""
        pattern_dict = {
            'Personal Information': [
                r'(?:please\s+)?state\s+your\s+(?:full\s+)?name',
                r'what\s+is\s+your\s+name',
                r'where\s+do\s+you\s+live',
                r'(?:current\s+)?address'
            ],
            'Employment': [
                r'where\s+do\s+you\s+work',
                r'(?:current\s+)?employer',
                r'what\s+is\s+your\s+(?:job|occupation)',
                r'employment\s+history'
            ],
            'Medical History': [
                r'medical\s+history',
                r'(?:any\s+)?(?:prior\s+)?surgeries',
                r'health\s+conditions',
                r'medications'
            ],
            'Incident Details': [
                r'(?:day|date)\s+of\s+the\s+(?:accident|incident)',
                r'tell\s+us\s+what\s+happened',
                r'describe\s+the\s+(?:accident|incident)'
            ],
            'Current Condition': [
                r'current\s+condition',
                r'how\s+are\s+you\s+feeling',
                r'(?:still\s+)?treating',
                r'pain\s+level'
            ]
        }
        
        compiled = {}
        for topic, patterns in pattern_dict.items():
            compiled[topic] = []
            for pattern in patterns:
                try:
                    compiled[topic].append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    st.warning(f"Invalid regex pattern for {topic}: {pattern} - {e}")
                    continue
        
        return compiled

    def chunk_transcript(self, transcript_text: str) -> List[str]:
        """Main method with improved reliability and performance"""
        start_time = time.time()
        
        # Enhanced input validation
        if not transcript_text or not transcript_text.strip():
            st.warning("Empty or whitespace-only text provided")
            return []
            
        text_length = len(transcript_text.strip())
        self.processing_stats['total_text_length'] = text_length
        
        if text_length < self.config.min_text_length:
            st.warning(f"Text too short for topic-based chunking (min: {self.config.min_text_length})")
            return self._simple_chunk(transcript_text)

        try:
            st.info("Analyzing transcript structure...")
            topic_segments = self._multi_method_topic_detection(transcript_text)

            if not topic_segments:
                st.warning("No topic segments detected, using simple chunking")
                return self._simple_chunk(transcript_text)

            quality_score = self._evaluate_segmentation_quality(topic_segments, text_length)
            st.info(f"Segmentation quality score: {quality_score:.2f}")

            if quality_score < self.config.quality_threshold:
                st.warning("Applying segmentation corrections...")
                topic_segments = self._correct_segments_safe(topic_segments)

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

    def _get_text_hash(self, text: str) -> str:
        """Generate hash for caching"""
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def _multi_method_topic_detection(self, text: str) -> List[TopicSegment]:
        """Use multiple methods and combine results with caching"""
        text_hash = self._get_text_hash(text) if self.enable_caching else None
        cache_key = f"detection_{text_hash}" if text_hash else None
        
        # Check cache first
        if cache_key and cache_key in self._response_cache:
            self.processing_stats['cache_hits'] += 1
            return self._response_cache[cache_key]
        
        all_segments = []
        
        # Method 1: LLM-based detection (if available)
        if self.call_llm:
            try:
                llm_segments = self._identify_topics_llm_optimized(text)
                if llm_segments and self._validate_segments(llm_segments):
                    all_segments.extend(llm_segments)
            except Exception as e:
                st.warning(f"LLM detection failed: {e}")
        
        # Method 2: Pattern-based detection (always available)
        try:
            pattern_segments = self._pattern_based_segmentation_enhanced(text)
            if pattern_segments and self._validate_segments(pattern_segments):
                if not all_segments:
                    all_segments = pattern_segments
                else:
                    all_segments = self._merge_detection_results(all_segments, pattern_segments)
        except Exception as e:
            st.warning(f"Pattern detection failed: {e}")
        
        # Fallback if all methods failed
        if not all_segments:
            st.warning("All detection methods failed, using uniform segments")
            all_segments = self._create_uniform_segments(text)
        
        # Cache the result
        if cache_key:
            self._response_cache[cache_key] = all_segments
        
        return all_segments

    def _validate_segments(self, segments: List[TopicSegment]) -> bool:
        """Validate that segments are reasonable"""
        if not segments:
            return False
        
        # Check for overlapping segments
        for i in range(len(segments) - 1):
            if segments[i].end_idx > segments[i + 1].start_idx:
                st.warning(f"Overlapping segments detected: {i} and {i+1}")
                return False
        
        # Check coverage
        total_coverage = sum(seg.end_idx - seg.start_idx for seg in segments)
        if total_coverage < 0.8 * (segments[-1].end_idx - segments[0].start_idx):
            st.warning("Poor text coverage in segments")
            return False
        
        return True

    def _identify_topics_llm_optimized(self, text: str) -> List[TopicSegment]:
        """Optimized LLM-based topic identification with better caching"""
        if not self.call_llm:
            raise Exception("LLM function not available")
        
        # Create cache key for LLM response
        text_hash = self._get_text_hash(text) if self.enable_caching else None
        llm_cache_key = f"llm_{self.llm_provider}_{self.llm_model}_{text_hash}" if text_hash else None
        
        # Check LLM cache
        if llm_cache_key and llm_cache_key in self._response_cache:
            self.processing_stats['cache_hits'] += 1
            response = self._response_cache[llm_cache_key]
        else:
            # Strategic text sampling
            text_length = len(text)
            sample_size = min(15000, text_length)
            
            samples = self._get_strategic_samples(text, sample_size)
            
            # Improved prompt with better structure
            prompt = self._create_llm_prompt(samples)
            
            self.processing_stats['llm_calls'] += 1
            response = self.call_llm(prompt, self.llm_provider, self.llm_model)
            
            if response.startswith("Error:"):
                raise Exception(f"LLM call failed: {response}")
            
            # Cache the response
            if llm_cache_key:
                self._response_cache[llm_cache_key] = response
        
        return self._parse_llm_response_enhanced(response, text)

    def _get_strategic_samples(self, text: str, sample_size: int) -> List[str]:
        """Get strategic samples from text for LLM analysis"""
        text_length = len(text)
        
        if text_length <= sample_size:
            return [text]
        
        # Get samples from beginning, middle, and end
        third = sample_size // 3
        samples = [
            text[:third],
            text[text_length//2 - third//2:text_length//2 + third//2],
            text[-third:]
        ]
        
        return samples

    def _create_llm_prompt(self, samples: List[str]) -> str:
        """Create structured prompt for LLM"""
        return f"""
Analyze this deposition transcript and identify topic transitions with high precision.

COMMON DEPOSITION TOPICS:
1. Personal Information (name, address, background)
2. Employment History (current/past jobs, responsibilities)
3. Medical History (pre-incident health, medications)
4. The Incident/Accident (what happened, when, where)
5. Post-Incident Events (immediate aftermath)
6. Medical Treatment (doctors, procedures, therapy)
7. Current Condition (ongoing symptoms, limitations)

REQUIREMENTS:
- Identify clear topic transitions with unique text markers
- Provide confidence scores based on clarity of transition
- Use exact phrases from the text as markers

FORMAT (one per line):
TOPIC: [exact topic name] | MARKER: [unique phrase from text] | CONFIDENCE: [0.0-1.0]

TRANSCRIPT SAMPLES:
{' [...NEXT SAMPLE...] '.join(samples)}
"""

    def _parse_llm_response_enhanced(self, response: str, full_text: str) -> List[TopicSegment]:
        """Enhanced parsing with better validation"""
        segments = []
        lines = response.split('\n')
        markers = []
        
        # Extract and validate markers
        for line_num, line in enumerate(lines):
            if 'TOPIC:' in line and 'MARKER:' in line:
                try:
                    marker_data = self._parse_marker_line(line)
                    if marker_data and self._validate_marker(marker_data, full_text):
                        markers.append(marker_data)
                except Exception as e:
                    st.debug(f"Failed to parse line {line_num}: {line} - {e}")
                    continue
        
        if not markers:
            st.warning("No valid markers found in LLM response")
            return []
        
        # Create segments with improved positioning
        return self._create_segments_from_markers(markers, full_text)

    def _parse_marker_line(self, line: str) -> Optional[Dict]:
        """Parse a single marker line from LLM response"""
        parts = line.split('|')
        if len(parts) < 2:
            return None
        
        marker_data = {'topic': '', 'marker': '', 'confidence': 0.7}
        
        for part in parts:
            part = part.strip()
            if part.startswith('TOPIC:'):
                marker_data['topic'] = part[6:].strip()
            elif part.startswith('MARKER:'):
                marker_data['marker'] = part[7:].strip()
            elif part.startswith('CONFIDENCE:'):
                try:
                    confidence_str = part[11:].strip()
                    marker_data['confidence'] = max(0.0, min(1.0, float(confidence_str)))
                except (ValueError, TypeError):
                    marker_data['confidence'] = 0.7
        
        return marker_data if marker_data['topic'] and marker_data['marker'] else None

    def _validate_marker(self, marker_data: Dict, text: str) -> bool:
        """Validate that marker exists in text and is reasonable"""
        marker = marker_data['marker']
        if len(marker) < 5 or len(marker) > 200:  # Reasonable marker length
            return False
        
        # Check if marker exists in text (fuzzy)
        if marker.lower() not in text.lower():
            # Try partial match
            words = marker.split()
            if len(words) >= 2:
                partial = ' '.join(words[:2])
                if partial.lower() not in text.lower():
                    return False
            else:
                return False
        
        return True

    def _create_segments_from_markers(self, markers: List[Dict], full_text: str) -> List[TopicSegment]:
        """Create segments from validated markers"""
        segments = []
        last_pos = 0
        
        for i, marker_info in enumerate(markers):
            marker_pos = self._find_marker_position(full_text, marker_info['marker'], last_pos)
            
            if marker_pos != -1 and marker_pos >= last_pos:
                # Create segment from last position to this marker
                if i > 0:  # Not the first marker
                    prev_topic = markers[i-1]['topic']
                    prev_confidence = markers[i-1]['confidence']
                else:
                    prev_topic = 'Introduction'
                    prev_confidence = 0.6
                
                if marker_pos > last_pos:  # Avoid empty segments
                    segments.append(TopicSegment(
                        topic=prev_topic,
                        start_idx=last_pos,
                        end_idx=marker_pos,
                        text=full_text[last_pos:marker_pos],
                        confidence=prev_confidence,
                        method='llm'
                    ))
                
                last_pos = marker_pos
        
        # Add final segment
        if last_pos < len(full_text) and markers:
            final_marker = markers[-1]
            segments.append(TopicSegment(
                topic=final_marker['topic'],
                start_idx=last_pos,
                end_idx=len(full_text),
                text=full_text[last_pos:],
                confidence=final_marker['confidence'],
                method='llm'
            ))
        
        return segments

    def _find_marker_position(self, text: str, marker: str, start_pos: int = 0) -> int:
        """Find marker position with fuzzy matching"""
        # Try exact match first
        pos = text.find(marker, start_pos)
        if pos != -1:
            return pos
        
        # Try case-insensitive
        pos = text.lower().find(marker.lower(), start_pos)
        if pos != -1:
            return pos
        
        # Try progressive word matching
        words = marker.split()
        for num_words in range(len(words), max(1, len(words)//2), -1):
            partial = ' '.join(words[:num_words])
            pos = text.lower().find(partial.lower(), start_pos)
            if pos != -1:
                return pos
        
        return -1

    def _pattern_based_segmentation_enhanced(self, text: str) -> List[TopicSegment]:
        """Enhanced pattern matching with compiled regex"""
        found_positions = []
        
        # Use pre-compiled patterns for better performance
        for topic, compiled_patterns in self._compiled_patterns.items():
            for pattern in compiled_patterns:
                try:
                    matches = list(pattern.finditer(text))
                    for match in matches:
                        confidence = self._calculate_pattern_confidence(match, text)
                        found_positions.append({
                            'topic': topic,
                            'position': match.start(),
                            'text': match.group(),
                            'confidence': confidence
                        })
                except Exception as e:
                    st.debug(f"Pattern matching error for {topic}: {e}")
                    continue
        
        # Sort by position and remove duplicates
        found_positions.sort(key=lambda x: x['position'])
        found_positions = self._remove_duplicate_positions(found_positions)
        
        return self._create_segments_from_positions(found_positions, text)

    def _calculate_pattern_confidence(self, match, text: str) -> float:
        """Calculate confidence based on pattern context"""
        base_confidence = 0.7
        
        # Increase confidence if match is at sentence boundary
        start = match.start()
        if start == 0 or text[start-1] in '.!?\n':
            base_confidence += 0.1
        
        # Increase confidence if followed by question structure
        end = match.end()
        following_text = text[end:end+50].lower()
        if any(word in following_text for word in ['?', 'please', 'tell', 'describe']):
            base_confidence += 0.1
        
        return min(1.0, base_confidence)

    def _remove_duplicate_positions(self, positions: List[Dict]) -> List[Dict]:
        """Remove positions that are too close together"""
        if not positions:
            return []
        
        filtered = [positions[0]]
        min_gap = self.config.min_segment_gap
        
        for pos in positions[1:]:
            if pos['position'] - filtered[-1]['position'] >= min_gap:
                filtered.append(pos)
        
        return filtered

    def _create_segments_from_positions(self, positions: List[Dict], text: str) -> List[TopicSegment]:
        """Create segments from position data"""
        if not positions:
            return []
        
        segments = []
        last_pos = 0
        
        for i, pos_info in enumerate(positions):
            if pos_info['position'] > last_pos:
                # Determine topic for this segment
                if i > 0:
                    prev_topic = positions[i-1]['topic']
                    prev_confidence = positions[i-1]['confidence']
                else:
                    prev_topic = 'Introduction'
                    prev_confidence = 0.6
                
                segments.append(TopicSegment(
                    topic=prev_topic,
                    start_idx=last_pos,
                    end_idx=pos_info['position'],
                    text=text[last_pos:pos_info['position']],
                    confidence=prev_confidence,
                    method='pattern'
                ))
                
                last_pos = pos_info['position']
        
        # Add final segment
        if last_pos < len(text):
            final_topic = positions[-1]['topic'] if positions else 'Content'
            final_confidence = positions[-1]['confidence'] if positions else 0.6
            
            segments.append(TopicSegment(
                topic=final_topic,
                start_idx=last_pos,
                end_idx=len(text),
                text=text[last_pos:],
                confidence=final_confidence,
                method='pattern'
            ))
        
        return segments

    def _create_uniform_segments(self, text: str) -> List[TopicSegment]:
        """Create uniform segments as fallback"""
        segments = []
        segment_size = self.config.target_chunk_size
        overlap = self.config.overlap_size
        
        for i in range(0, len(text), segment_size - overlap):
            end = min(i + segment_size, len(text))
            if end - i < self.config.min_chunk_size and segments:
                # Merge with last segment if too small
                segments[-1].text += text[i:end]
                segments[-1].end_idx = end
            else:
                segments.append(TopicSegment(
                    topic=f"Section {len(segments) + 1}",
                    start_idx=i,
                    end_idx=end,
                    text=text[i:end],
                    confidence=0.5,
                    method='uniform'
                ))
        
        return segments

    def _merge_detection_results(self, primary: List[TopicSegment], 
                               secondary: List[TopicSegment]) -> List[TopicSegment]:
        """Intelligently merge results from different detection methods"""
        if not secondary:
            return primary
        if not primary:
            return secondary
        
        # For now, prefer LLM results but could implement more sophisticated merging
        # TODO: Implement weighted merging based on confidence scores
        return primary

    def _evaluate_segmentation_quality(self, segments: List[TopicSegment], 
                                     total_length: int) -> float:
        """Enhanced quality evaluation"""
        if not segments:
            return 0.0
        
        quality_factors = []
        
        # 1. Size uniformity (target around target_chunk_size)
        sizes = [len(s.text) for s in segments]
        if len(sizes) > 1:
            target = self.config.target_chunk_size
            size_deviations = [abs(size - target) / target for size in sizes]
            avg_deviation = sum(size_deviations) / len(size_deviations)
            size_quality = max(0, 1 - avg_deviation)
            quality_factors.append(('size_uniformity', size_quality, 0.3))
        
        # 2. Confidence average
        avg_confidence = sum(s.confidence for s in segments) / len(segments)
        quality_factors.append(('confidence', avg_confidence, 0.3))
        
        # 3. Coverage (no significant gaps)
        segment_coverage = sum(s.end_idx - s.start_idx for s in segments)
        expected_coverage = segments[-1].end_idx - segments[0].start_idx if segments else 0
        coverage_ratio = min(1.0, segment_coverage / expected_coverage) if expected_coverage > 0 else 0
        quality_factors.append(('coverage', coverage_ratio, 0.2))
        
        # 4. Segment count appropriateness
        expected_segments = max(1, total_length // self.config.target_chunk_size)
        actual_segments = len(segments)
        segment_ratio = min(actual_segments / expected_segments, expected_segments / actual_segments) if expected_segments > 0 else 0
        quality_factors.append(('segment_count', segment_ratio, 0.2))
        
        # Calculate weighted average
        total_weight = sum(weight for _, _, weight in quality_factors)
        weighted_score = sum(score * weight for _, score, weight in quality_factors) / total_weight
        
        return weighted_score

    def _correct_segments_safe(self, segments: List[TopicSegment]) -> List[TopicSegment]:
        """Safe segment correction without list modification during iteration"""
        if not segments:
            return segments
        
        corrected = []
        skip_next = False
        
        for i in range(len(segments)):
            if skip_next:
                skip_next = False
                continue
            
            current = segments[i]
            
            # Skip empty segments
            if not current.text.strip():
                continue
            
            # Try to merge small segments with next segment
            if (len(current.text) < self.config.min_chunk_size and 
                i < len(segments) - 1):
                
                next_segment = segments[i + 1]
                combined_size = len(current.text) + len(next_segment.text)
                
                if combined_size < self.config.max_chunk_size:
                    # Create merged segment
                    merged = TopicSegment(
                        topic=f"{current.topic} & {next_segment.topic}",
                        start_idx=current.start_idx,
                        end_idx=next_segment.end_idx,
                        text=current.text + "\n\n" + next_segment.text,
                        confidence=min(current.confidence, next_segment.confidence),
                        method='corrected-merged'
                    )
                    corrected.append(merged)
                    skip_next = True  # Skip the next segment since we merged it
                    continue
            
            # Add segment as-is if no merging occurred
            corrected.append(current)
        
        return corrected

    def _optimize_segment_sizes(self, segments: List[TopicSegment]) -> List[TopicSegment]:
        """Optimize segment sizes for target range"""
        optimized = []
        
        # First pass: handle oversized segments
        for segment in segments:
            if len(segment.text) > self.config.max_chunk_size:
                splits = self._split_large_segment_enhanced(segment)
                optimized.extend(splits)
            else:
                optimized.append(segment)
        
        # Second pass: merge undersized segments
        final = []
        i = 0
        while i < len(optimized):
            current = optimized[i]
            
            # Try to merge small consecutive segments
            while (i < len(optimized) - 1 and 
                   len(current.text) < self.config.min_chunk_size):
                next_seg = optimized[i + 1]
                combined_size = len(current.text) + len(next_seg.text)
                
                if combined_size <= self.config.max_chunk_size:
                    current = TopicSegment(
                        topic=f"{current.topic} + {next_seg.topic}",
                        start_idx=current.start_idx,
                        end_idx=next_seg.end_idx,
                        text=current.text + "\n\n" + next_seg.text,
                        confidence=min(current.confidence, next_seg.confidence),
                        method='merged'
                    )
                    i += 1
                else:
                    break
            
            final.append(current)
            i += 1
        
        return final

    def _split_large_segment_enhanced(self, segment: TopicSegment) -> List[TopicSegment]:
        """Enhanced splitting with better boundary detection"""
        if len(segment.text) <= self.config.max_chunk_size:
            return [segment]
        
        splits = []
        text = segment.text
        target_size = self.config.target_chunk_size
        
        # Try to split at paragraph boundaries first
        paragraphs = text.split('\n\n')
        
        if len(paragraphs) > 1:
            splits = self._split_by_paragraphs(paragraphs, segment, target_size)
        else:
            # Fall back to sentence splitting
            splits = self._split_by_sentences(text, segment, target_size)
        
        return splits if splits else [segment]

    def _split_by_paragraphs(self, paragraphs: List[str], 
                           segment: TopicSegment, target_size: int) -> List[TopicSegment]:
        """Split segment by paragraph boundaries"""
        splits = []
        current_text = ""
        split_count = 0
        
        for para in paragraphs:
            if len(current_text) + len(para) <= target_size:
                current_text += para + "\n\n" if current_text else para
            else:
                if current_text:
                    splits.append(self._create_split_segment(
                        segment, current_text.strip(), split_count
                    ))
                    split_count += 1
                current_text = para
        
        # Add remaining text
        if current_text:
            splits.append(self._create_split_segment(
                segment, current_text.strip(), split_count
            ))
        
        return splits

    def _split_by_sentences(self, text: str, 
                          segment: TopicSegment, target_size: int) -> List[TopicSegment]:
        """Split segment by sentence boundaries"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        splits = []
        current_text = ""
        split_count = 0
        
        for sentence in sentences:
            if len(current_text) + len(sentence) <= target_size:
                current_text += " " + sentence if current_text else sentence
            else:
                if current_text:
                    splits.append(self._create_split_segment(
                        segment, current_text.strip(), split_count
                    ))
                    split_count += 1
                current_text = sentence
        
        # Add remaining text
        if current_text:
            splits.append(self._create_split_segment(
                segment, current_text.strip(), split_count
            ))
        
        return splits

    def _create_split_segment(self, original: TopicSegment, 
                            text: str, split_index: int) -> TopicSegment:
        """Create a split segment with proper metadata"""
        return TopicSegment(
            topic=f"{original.topic} (Part {split_index + 1})",
            start_idx=original.start_idx,  # Approximate
            end_idx=original.start_idx + len(text),  # Approximate
            text=text,
            confidence=original.confidence * 0.95,  # Slight confidence reduction
            method='split'
        )

    def _create_final_chunks_enhanced(self, segments: List[TopicSegment]) -> List[str]:
        """Create final chunks with enhanced overlap and metadata"""
        if not segments:
            return []
        
        final_chunks = []
        
        for i, segment in enumerate(segments):
            chunk_parts = []
            
            # Add metadata header
            chunk_parts.append(f"[TOPIC: {segment.topic}]")
            chunk_parts.append(f"[Method: {segment.method}, Confidence: {segment.confidence:.2f}]")
            
            # Add context from previous segment (safe overlap)
            if i > 0 and self.config.overlap_size > 0:
                prev_text = segments[i-1].text
                if prev_text:
                    overlap_text = self._extract_safe_overlap(
                        prev_text, self.config.overlap_size, is_prefix=False
                    )
                    if overlap_text:
                        chunk_parts.append(f"\n[CONTEXT: ...{overlap_text}]\n")
            
            # Add main content
            chunk_parts.append(segment.text)
            
            # Add preview of next segment (safe preview)
            if i < len(segments) - 1 and self.config.overlap_size > 0:
                next_text = segments[i+1].text
                if next_text:
                    preview_text = self._extract_safe_overlap(
                        next_text, self.config.overlap_size, is_prefix=True
                    )
                    if preview_text:
                        chunk_parts.append(f"\n[CONTINUES: {preview_text}...]\n")
            
            final_chunks.append("\n".join(chunk_parts))
        
        return final_chunks

    def _extract_safe_overlap(self, text: str, max_size: int, is_prefix: bool) -> str:
        """Safely extract overlap text with sentence boundary detection"""
        if not text or max_size <= 0:
            return ""
        
        if is_prefix:
            # Extract from beginning
            if len(text) <= max_size:
                return text
            
            extract = text[:max_size]
            # Find last sentence boundary
            match = re.search(r'^(.*?[.!?])', extract)
            if match and len(match.group(1)) >= self.config.preview_sentence_min_length:
                return match.group(1)
            else:
                # Fall back to word boundary
                words = extract.split()
                return ' '.join(words[:-1]) if len(words) > 1 else extract
        else:
            # Extract from end
            if len(text) <= max_size:
                return text
            
            extract = text[-max_size:]
            # Find first sentence boundary
            sentences = re.split(r'(?<=[.!?])\s+', extract)
            if len(sentences) > 1 and len(sentences[-1]) >= self.config.preview_sentence_min_length:
                return sentences[-1]
            else:
                # Fall back to word boundary
                words = extract.split()
                return ' '.join(words[1:]) if len(words) > 1 else extract

    def _validate_final_chunks(self, chunks: List[str]) -> List[str]:
        """Enhanced final validation with better error recovery"""
        if not chunks:
            return []
        
        validated = []
        
        for i, chunk in enumerate(chunks):
            # Remove empty or tiny chunks
            if not chunk or len(chunk.strip()) < 50:
                st.debug(f"Removing tiny chunk {i}: {len(chunk)} chars")
                continue
            
            # Handle oversized chunks
            if len(chunk) > self.config.max_chunk_size * self.config.emergency_split_threshold:
                st.warning(f"Emergency splitting oversized chunk {i}: {len(chunk)} chars")
                # Emergency split at midpoint
                mid = len(chunk) // 2
                # Try to split at sentence boundary near midpoint
                nearby_text = chunk[mid-100:mid+100]
                sentence_match = re.search(r'[.!?]\s+', nearby_text)
                if sentence_match:
                    split_pos = mid - 100 + sentence_match.end()
                    validated.append(chunk[:split_pos])
                    validated.append(chunk[split_pos:])
                else:
                    validated.append(chunk[:mid])
                    validated.append(chunk[mid:])
            else:
                validated.append(chunk)
        
        # Ensure we have at least one chunk
        if not validated and chunks:
            validated = [chunks[0]]
        
        return validated

    @lru_cache(maxsize=100)
    def _simple_chunk(self, text: str) -> List[str]:
        """Cached simple fallback chunking"""
        if not text:
            return []
        
        chunks = []
        target_size = self.config.target_chunk_size
        overlap = self.config.overlap_size
        
        for i in range(0, len(text), target_size - overlap):
            chunk = text[i:i + target_size]
            if chunk and len(chunk.strip()) > 10:
                chunks.append(chunk)
        
        return chunks if chunks else [text]

    def get_stats(self) -> Dict:
        """Get enhanced processing statistics"""
        stats = self.processing_stats.copy()
        if self._response_cache:
            stats['cache_size'] = len(self._response_cache)
        return stats

    def clear_cache(self):
        """Clear the response cache"""
        if self._response_cache:
            self._response_cache.clear()
            st.info("Cache cleared")

    def get_config(self) -> ChunkerConfig:
        """Get current configuration"""
        return self.config

    def update_config(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                st.warning(f"Unknown configuration parameter: {key}")