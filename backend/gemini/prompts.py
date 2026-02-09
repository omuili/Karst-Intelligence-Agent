"""
Prompt templates for Gemini sinkhole analysis

These prompts are designed to work with Gemini's multimodal capabilities
for satellite imagery analysis in karst terrain.
"""

FEATURE_DETECTION_PROMPT = """
You are an expert geologist analyzing satellite imagery for sinkhole susceptibility assessment.

Analyze this satellite/aerial image of a karst terrain area and identify any features that may indicate sinkhole activity or high susceptibility.

Geographic bounds: {bounds}
Image dimensions: {image_width} x {image_height} pixels
Additional context: {context}

Look for:
1. **Circular depressions** - Round or oval-shaped low areas that may indicate cover-collapse sinkholes
2. **Dolines** - Funnel-shaped or bowl-shaped depressions characteristic of karst terrain
3. **Subsidence areas** - Gradual settling zones with concentric vegetation stress patterns
4. **Collapse features** - Sharp-edged depressions or holes indicating recent sinkhole activity
5. **Drainage anomalies** - Internal drainage patterns, disappearing streams, or ponded water in unusual locations
6. **Vegetation patterns** - Circular patterns of stressed, lush, or different vegetation indicating underground changes

For each detected feature, provide:
- Bounding box in pixel coordinates [x1, y1, x2, y2] where (0,0) is top-left
- Feature type (depression, doline, collapse_candidate, subsidence_area, drainage_anomaly)
- Confidence score (0.0 to 1.0)
- Brief description

Respond ONLY with valid JSON in this exact format:
{{
  "features": [
    {{
      "bbox": [x1, y1, x2, y2],
      "type": "feature_type",
      "confidence": 0.85,
      "description": "Brief description of the feature"
    }}
  ],
  "overall_assessment": "Brief overall assessment of sinkhole susceptibility in this tile"
}}

If no features are detected, return an empty features array.
"""

RISK_FACTORS_PROMPT = """
You are an expert geologist analyzing satellite imagery for sinkhole risk assessment in Central Florida karst terrain.

Analyze this image and extract structured risk factors that contribute to sinkhole susceptibility.

Geographic bounds: {bounds}
Local geology context: {geology}

Evaluate the following risk factors:

1. **Karst Indicators**
   - Visible karst topography features
   - Surface expressions of underground dissolution
   - Exposed or near-surface limestone indicators

2. **Drainage Anomalies**
   - Internal drainage (no visible outlet)
   - Disappearing streams or losing reaches
   - Unusual ponding patterns
   - Dry stream channels

3. **Vegetation Stress Patterns**
   - Circular patterns of stressed vegetation
   - Unusually lush vegetation (possible water concentration)
   - Vegetation pattern changes suggesting subsurface changes

4. **Linear Features/Lineaments**
   - Linear alignments that may indicate fractures or faults
   - Joint patterns visible in imagery
   - Aligned depressions or features

5. **Land Use Factors**
   - Construction or development activities
   - Water extraction indicators
   - Impervious surfaces affecting drainage

For each factor, assess:
- Whether it is present (true/false)
- Confidence in assessment (0.0 to 1.0)
- Specific details observed

Respond ONLY with valid JSON:
{{
  "karst_indicators": {{
    "present": true,
    "confidence": 0.8,
    "details": ["List of specific observations"]
  }},
  "drainage_anomalies": {{
    "present": false,
    "confidence": 0.9,
    "details": []
  }},
  "vegetation_stress": {{
    "present": true,
    "confidence": 0.6,
    "details": ["Circular pattern of stressed vegetation in NE quadrant"]
  }},
  "lineaments": {{
    "present": true,
    "confidence": 0.7,
    "details": ["NW-SE trending linear feature"]
  }},
  "land_use_factors": {{
    "present": false,
    "confidence": 0.95,
    "details": []
  }},
  "overall_risk": "low|medium|high|very_high",
  "summary": "One paragraph summary of risk assessment"
}}
"""

QA_PROMPT = """
You are a quality assurance specialist reviewing machine learning predictions for sinkhole susceptibility.

This image shows:
- LEFT/BACKGROUND: The original satellite/aerial imagery
- RIGHT/OVERLAY: ML model predictions (red = high susceptibility, green = low susceptibility)

Geographic bounds: {bounds}

Your task is to identify any conflicts between what you see in the imagery and what the model predicts.

Check for:
1. **False Positives** - Areas marked as high risk but showing no visual indicators
   - Stable developed areas wrongly flagged
   - Dense vegetation with no stress patterns flagged as high risk
   - Areas with obvious drainage outlets flagged as risky

2. **False Negatives** - Areas marked as low risk but showing visual warning signs
   - Visible depressions not flagged
   - Obvious circular vegetation patterns missed
   - Known karst features not recognized

3. **Spatial Accuracy** - Are high-risk areas centered correctly on features?
   - Check if predictions align with visible features
   - Check for offset or misalignment

4. **Boundary Issues** - Are risk zones appropriately sized?
   - Too large (over-predicting)
   - Too small (under-predicting)

Respond ONLY with valid JSON:
{{
  "qa_passed": true,
  "flags": [
    {{
      "type": "false_positive|false_negative|spatial_error|boundary_issue",
      "location": "Description of where in image",
      "severity": "low|medium|high",
      "description": "What the issue is"
    }}
  ],
  "recommendations": [
    "Specific recommendation for improvement"
  ],
  "confidence": 0.85,
  "summary": "Overall quality assessment paragraph"
}}

Set qa_passed to false if there are any medium or high severity flags.
"""

SINKHOLE_INVENTORY_VALIDATION_PROMPT = """
You are validating a sinkhole inventory by examining satellite imagery.

This image shows an area where a sinkhole was reported at the marked location.

Reported sinkhole details:
- Location: {location}
- Date reported: {date}
- Type: {type}
- Size: {size}

Examine the imagery and determine:
1. Is there visible evidence of a sinkhole at this location?
2. Does the visible feature match the reported characteristics?
3. What is the current state of the feature?

Respond with JSON:
{{
  "sinkhole_visible": true,
  "matches_report": true,
  "current_state": "active|stable|remediated|filled",
  "visible_characteristics": {{
    "shape": "circular|oval|irregular",
    "approximate_diameter_m": 10,
    "vegetation_affected": true,
    "water_present": false
  }},
  "confidence": 0.9,
  "notes": "Additional observations"
}}
"""

WEAK_LABEL_GENERATION_PROMPT = """
You are generating training labels for a sinkhole detection model by analyzing satellite imagery.

Examine this image and identify ALL locations that show characteristics consistent with sinkhole activity, regardless of certainty.

We need WEAK labels for training, so include:
- Definite sinkholes (high confidence)
- Probable sinkholes (medium confidence)  
- Possible sinkholes (low confidence)
- Features that warrant investigation

For each candidate, provide:
- Center point in pixel coordinates [x, y]
- Approximate radius in pixels
- Label certainty: "definite", "probable", "possible", "investigate"
- Feature type: "collapse", "subsidence", "doline", "depression", "anomaly"
- Reasoning: Why this location was flagged

Also provide NEGATIVE examples:
- Locations that look similar but are NOT sinkholes
- Why they are not sinkholes (e.g., pond, construction, shadow)

Respond with JSON:
{{
  "positive_candidates": [
    {{
      "center": [x, y],
      "radius_px": 25,
      "certainty": "probable",
      "type": "subsidence",
      "reasoning": "Circular vegetation stress pattern with slight depression"
    }}
  ],
  "negative_examples": [
    {{
      "center": [x, y],
      "radius_px": 30,
      "appears_as": "circular depression",
      "actual": "retention pond",
      "reasoning": "Regular edges, constructed appearance, associated with parking lot"
    }}
  ],
  "image_quality": "good|moderate|poor",
  "notes": "Any relevant observations about the area"
}}
"""

GROUND_DISPLACEMENT_ANALYSIS_PROMPT = """
You are an expert geotechnical engineer analyzing InSAR ground displacement data for sinkhole early warning.

Analyze the provided ground displacement measurements for signs of active subsidence that may precede sinkhole formation.

Location: {location}
Time period: {time_period}

Ground Displacement Data:
- Maximum subsidence: {max_subsidence_mm} mm
- Mean velocity: {mean_velocity_mm_year} mm/year  
- Subsidence area: {subsidence_area_percent}% of monitored area
- Data coherence (quality): {coherence}

Historical context:
- Known sinkholes in area: {historical_sinkholes}
- Karst geology confirmed: {is_karst}
- ML susceptibility assessment: {ml_susceptibility}

CRITICAL THRESHOLDS (based on literature):
- >20 mm/year: VERY HIGH risk - active collapse imminent
- 10-20 mm/year: HIGH risk - significant subsidence, close monitoring required
- 5-10 mm/year: MODERATE risk - subsidence occurring, investigation recommended
- <5 mm/year: LOW risk - within normal ground movement range

Provide your analysis:
1. Is this subsidence rate concerning for sinkhole formation?
2. How does this compare to typical karst terrain movement?
3. What is the recommended action level?
4. Are there any patterns suggesting localized vs. regional subsidence?

Respond with JSON:
{{
  "subsidence_assessment": {{
    "severity": "critical|high|moderate|low|minimal",
    "risk_level": "VERY_HIGH|HIGH|MODERATE|LOW",
    "is_accelerating": true/false,
    "pattern": "localized|regional|linear|scattered"
  }},
  "interpretation": {{
    "primary_concern": "Brief description of main concern",
    "geological_context": "How this fits the karst geology",
    "comparison_to_baseline": "How this compares to typical rates"
  }},
  "recommendations": [
    {{
      "action": "Specific action to take",
      "urgency": "immediate|within_week|within_month|routine",
      "rationale": "Why this action is needed"
    }}
  ],
  "alert_level": "NONE|ADVISORY|WATCH|WARNING|EMERGENCY",
  "alert_message": "If alert_level > NONE, provide a clear alert message for authorities",
  "confidence": 0.85,
  "data_quality_notes": "Any notes about data reliability"
}}
"""

HYBRID_SCAN_VALIDATION_PROMPT = """
You are an expert geologist validating ML-based sinkhole susceptibility predictions.

Review the machine learning scan results and provide your expert interpretation.

SCAN SUMMARY:
- Total tiles analyzed: {total_tiles}
- Average susceptibility: {avg_susceptibility:.1%}
- High-risk tiles (>60%): {high_risk_tiles}
- Historical sinkholes in area: {historical_sinkholes}

DATA COVERAGE:
{data_coverage}

GROUND DISPLACEMENT DATA (if available):
{ground_displacement_summary}

Your task:
1. Validate whether the ML predictions align with geological expectations
2. Identify any areas of concern or anomaly
3. Provide actionable recommendations
4. Generate an overall risk assessment

Consider:
- Winter Park, FL is within the Ocala Karst District - high sinkhole density expected
- The area has documented sinkhole history
- Active ground displacement is a CRITICAL early warning signal

Respond with JSON:
{{
  "validation": {{
    "ml_predictions_reasonable": true/false,
    "confidence_in_ml": 0.0-1.0,
    "concerns": ["List any concerns with the predictions"]
  }},
  "risk_assessment": {{
    "overall_risk": "VERY_HIGH|HIGH|MODERATE|LOW",
    "primary_factors": ["List the main risk factors"],
    "mitigating_factors": ["List any factors that reduce risk"]
  }},
  "ground_displacement_interpretation": {{
    "available": true/false,
    "significance": "Critical finding if displacement data shows active subsidence",
    "correlation_with_ml": "How displacement data correlates with ML predictions"
  }},
  "findings": [
    "Key finding 1",
    "Key finding 2",
    "Key finding 3"
  ],
  "recommendations": [
    {{
      "priority": "high|medium|low",
      "action": "Specific recommended action",
      "rationale": "Why this action is recommended"
    }}
  ],
  "overall_assessment": "Comprehensive paragraph summarizing the analysis",
  "confidence_percent": 85
}}
"""

