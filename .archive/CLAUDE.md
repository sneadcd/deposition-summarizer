# Deposition Summarizer Enhancement Action Plan

## Overview
This action plan outlines improvements to the deposition summarizer based on code review findings. The focus is on handling lengthy deposition transcripts (up to 250 pages / 50k tokens) more reliably.

## Priority Improvements

### 1. Fix Anthropic Prompt Splitting Logic ✅ High Priority
**Issue**: Current code only splits on exact "BEGIN TRANSCRIPT" but prompts use variations like "BEGIN TRANSCRIPT TEXT:" and "BEGIN TRANSCRIPT SEGMENT:"

**Files to modify**: `app.py`

**Implementation**:
```python
# In app.py, inside call_llm function (~line 194)
# Replace existing split logic with:
import re  # Add at top if not already

# Inside call_llm, for Anthropic:
system_prompt, user_prompt = "", prompt
match = re.search(r'BEGIN TRANSCRIPT\s*(?:TEXT|SEGMENT)?\s*:?', prompt, flags=re.IGNORECASE)
if match:
    system_prompt = prompt[:match.start()].strip()
    user_prompt = prompt[match.start():].strip()
else:
    user_prompt = prompt
```

**Testing**: Process a deposition with Anthropic model and verify system/user prompts split correctly.

---

### 2. Add Dynamic Keyword Detection ✅ High Priority
**Issue**: Fixed keyword lists work for medical depositions but miss topics in corporate/criminal cases

**Files to modify**: `enhanced_topic_chunker_fixed.py`

**Implementation**:
```python
# In _multi_method_topic_detection, after LLM segments attempt (~line 225)
if not all_segments and self.call_llm:
    samples = self._get_strategic_samples(text, 15000)
    keyword_prompt = f"""Extract 5-8 main topics from this depo transcript samples, and for each topic list 3-5 associated keywords.
Format: TOPIC: keywords, separated, by, commas
Example: Employment History: job, position, duties, salary, employer

TRANSCRIPT SAMPLES:
{' [...] '.join(samples)}"""
    
    keyword_response = self.call_llm(keyword_prompt, self.llm_provider, self.llm_model)
    
    # Parse response
    dynamic_keywords = {}
    for line in keyword_response.split('\n'):
        if ':' in line:
            topic, kws = line.split(':', 1)
            dynamic_keywords[topic.strip()] = [kw.strip() for kw in kws.split(',')]
    
    # Update keyword sets
    for topic, keywords in dynamic_keywords.items():
        self.keyword_sets[topic] = set(kw.lower() for kw in keywords)
```

**Testing**: Run on depositions from different domains (medical, corporate, criminal) and verify topic detection improves.

---

### 3. Remove Cost Estimation Feature ✅ Medium Priority
**Issue**: Inaccurate and not worth the complexity

**Files to modify**: `app.py`

**Implementation**:
1. Delete `estimate_cost` function (lines ~347-362)
2. Remove cost estimation UI display (lines ~426-431)
3. Remove related session state variables

**Testing**: Verify UI still works without cost display.

---

### 4. Add Final Synthesis Batching Safety ✅ High Priority  
**Issue**: Combined segment summaries could exceed Claude's token limit on very long depositions

**Files to modify**: `app.py`

**Implementation**:
```python
# In Step 6: Final Synthesis (~line 517)
combined_summaries = "\n\n---\n\n".join([s for s in segment_summaries if not s.startswith("*** Error")])

# New: Batch if too long
max_prompt_chars = 150000  # Conservative for Claude
if len(combined_summaries) > max_prompt_chars:
    batch_size = 5  # Tune based on testing
    batched_summaries = []
    for i in range(0, len(segment_summaries), batch_size):
        batch = segment_summaries[i:i+batch_size]
        valid_batch = [s for s in batch if not s.startswith("*** Error")]
        if valid_batch:
            batch_text = "\n\n".join(valid_batch)
            batch_prompt = f"Summarize these segment summaries concisely:\n{batch_text}"
            batch_summary = call_llm_with_retry(batch_prompt, selected_provider_context, selected_model_context)
            if not batch_summary.startswith("Error:"):
                batched_summaries.append(batch_summary)
    
    combined_summaries = "\n\n---\n\n".join(batched_summaries)
```

**Testing**: Create synthetic test with 50+ segments to verify batching triggers correctly.

---

### 5. Enhance Chunker Sampling for Long Texts ✅ Medium Priority
**Issue**: Only samples 15k chars from beginning/middle/end, missing ~97% of very long depositions

**Files to modify**: `enhanced_topic_chunker_fixed.py`

**Implementation**:
```python
# Replace _get_strategic_samples (~line 277)
def _get_strategic_samples(self, text: str, sample_size: int) -> List[str]:
    text_length = len(text)
    
    if text_length <= sample_size:
        return [text]
    
    # Dynamic: More samples for longer texts
    if text_length < 50000:
        num_samples = 3
    elif text_length < 200000:
        num_samples = 5
    else:
        num_samples = 7
    
    sample_per = min(sample_size // num_samples, 5000)  # Cap individual samples
    samples = []
    
    for i in range(num_samples):
        start = int(i * text_length / num_samples)
        end = min(start + sample_per, text_length)
        samples.append(text[start:end])
    
    return samples
```

**Testing**: Test with depositions of varying lengths (50k, 200k, 500k chars) and verify topic detection quality.

---

## Additional Improvements (Lower Priority)

### 6. Add Cache Management
**Files to modify**: `enhanced_topic_chunker_fixed.py`
- Add max cache size limit
- Add method to clear cache between runs
- Log cache hit rate for performance monitoring

### 7. Improve Error Handling
**Files to modify**: `app.py`
- Filter error summaries more aggressively before final synthesis
- Add separate error log file
- Improve error messages with context

### 8. Add Progress Improvements
**Files to modify**: `app.py`
- Add estimated time remaining based on chunks processed
- Show which chunking method was used
- Display token counts for transparency

### 9. Make Chunker Config More Flexible
**Files to modify**: `app.py`
- Expose min_chunk_size, max_chunk_size in sidebar
- Add preset configs for different deposition types
- Save/load config settings

---

## Testing Plan

### Phase 1: Individual Feature Testing
- Test each improvement in isolation
- Use small test depositions first
- Verify no regressions

### Phase 2: Integration Testing  
- Test all improvements together
- Use variety of real depositions:
  - Short (10 pages)
  - Medium (50 pages)
  - Long (150+ pages)
  - Different types (medical, corporate, criminal)

### Phase 3: Edge Case Testing
- Empty/minimal text
- Corrupted formatting
- Mixed languages
- Unusual Q&A patterns

---

## Implementation Order

1. **Day 1**: Anthropic fix + Cost removal (quick wins)
2. **Day 2**: Synthesis batching + Enhanced sampling
3. **Day 3**: Dynamic keywords
4. **Day 4**: Testing and refinement
5. **Day 5**: Additional improvements as time permits

---

## Success Metrics

- ✅ No token limit errors on depositions up to 250 pages
- ✅ Successful processing of 95%+ of test depositions
- ✅ Topic detection works across different deposition types
- ✅ Clear error messages when issues occur
- ✅ Processing time reasonable (<30 min for longest depositions)

---

## Notes

- Keep original code backed up before changes
- Test with Gemini for context/segments, Claude for final synthesis
- Monitor API costs during testing phase
- Document any new issues discovered during implementation