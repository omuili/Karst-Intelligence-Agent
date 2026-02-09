const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        HeadingLevel, BorderStyle, WidthType, ShadingType, AlignmentType,
        LevelFormat, PageBreak } = require('docx');
const fs = require('fs');

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 80, bottom: 80, left: 120, right: 120 };

// Helper function to create a warning/note box as a table
function createNoteBox(text, type = "note") {
  const colors = {
    note: { fill: "EBF5FB", border: "2E86AB" },
    warning: { fill: "FDEDEC", border: "E74C3C" },
    tip: { fill: "E8F6F3", border: "27AE60" },
    important: { fill: "FEF9E7", border: "F39C12" }
  };
  const labels = {
    note: "NOTE",
    warning: "WARNING",
    tip: "TIP",
    important: "IMPORTANT"
  };
  const c = colors[type];
  const label = labels[type];
  
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: [
      new TableRow({
        children: [
          new TableCell({
            borders: {
              top: { style: BorderStyle.SINGLE, size: 12, color: c.border },
              bottom: { style: BorderStyle.SINGLE, size: 12, color: c.border },
              left: { style: BorderStyle.SINGLE, size: 12, color: c.border },
              right: { style: BorderStyle.SINGLE, size: 12, color: c.border }
            },
            shading: { fill: c.fill, type: ShadingType.CLEAR },
            margins: { top: 100, bottom: 100, left: 150, right: 150 },
            children: [
              new Paragraph({
                children: [
                  new TextRun({ text: label + ": ", bold: true, size: 22 }),
                  new TextRun({ text: text, size: 22 })
                ]
              })
            ]
          })
        ]
      })
    ]
  });
}

// Create the document
const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: "Arial", size: 22 }
      }
    },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: "1E3A5F" },
        paragraph: { spacing: { before: 400, after: 200 } }
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "2E5A8F" },
        paragraph: { spacing: { before: 350, after: 150 } }
      },
      {
        id: "Heading3",
        name: "Heading 3",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "3E6A9F" },
        paragraph: { spacing: { before: 300, after: 120 } }
      }
    ]
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0,
          format: LevelFormat.BULLET,
          text: "•",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      }
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    children: [
      // TITLE PAGE
      new Paragraph({ spacing: { after: 600 } }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "ABAQUS TUTORIAL", bold: true, size: 48, color: "1E3A5F" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 200, after: 400 },
        children: [new TextRun({ text: "2D Axisymmetric Analysis of Deeply Embedded Ring Anchor (DERA) Performance in Sand", bold: true, size: 32, color: "2E5A8F" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 600, after: 200 },
        children: [new TextRun({ text: "CORRECTED STEP-BY-STEP GUIDE", bold: true, size: 28, color: "E74C3C" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 400 },
        children: [new TextRun({ text: "For Abaqus Student/Learning Edition", size: 24, italics: true, color: "666666" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 800 },
        children: [new TextRun({ text: "This guide includes critical corrections for:", size: 22 })]
      }),
      new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "1. Soil Partitioning (Anchor Cavity)", size: 22, color: "27AE60" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "2. Mesh Gradation (Node Limit Compliance)", size: 22, color: "27AE60" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "3. Contact & Increment Settings (Convergence)", size: 22, color: "27AE60" })] }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // PART 1: INTRODUCTION
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("PART 1: Introduction & Overview")] }),
      
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("1.1 Project Description")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [
          new TextRun("This tutorial guides you through creating a 2D axisymmetric finite element model of a "),
          new TextRun({ text: "Deeply Embedded Ring Anchor (DERA)", bold: true }),
          new TextRun(" in sand using Abaqus. The model simulates anchor pullout behavior.")
        ]
      }),
      
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("1.2 Geometric Scaling (1:30)")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun("The actual DERA has diameter D = 3.0 m. This exceeds the Abaqus Student Edition 1000-node limit. We use 1:30 scaling:")]
      }),
      
      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        rows: [
          new TableRow({
            children: [
              new TableCell({ borders, shading: { fill: "2E5A8F", type: ShadingType.CLEAR }, margins: cellMargins, children: [new Paragraph({ children: [new TextRun({ text: "Parameter", bold: true, color: "FFFFFF" })] })] }),
              new TableCell({ borders, shading: { fill: "2E5A8F", type: ShadingType.CLEAR }, margins: cellMargins, children: [new Paragraph({ children: [new TextRun({ text: "Field Scale", bold: true, color: "FFFFFF" })] })] }),
              new TableCell({ borders, shading: { fill: "2E5A8F", type: ShadingType.CLEAR }, margins: cellMargins, children: [new Paragraph({ children: [new TextRun({ text: "Model Scale (1:30)", bold: true, color: "FFFFFF" })] })] })
            ]
          }),
          new TableRow({ children: [
            new TableCell({ borders, margins: cellMargins, children: [new Paragraph({ children: [new TextRun("Diameter (D)")] })] }),
            new TableCell({ borders, margins: cellMargins, children: [new Paragraph({ children: [new TextRun("3.0 m")] })] }),
            new TableCell({ borders, margins: cellMargins, children: [new Paragraph({ children: [new TextRun({ text: "0.10 m", bold: true })] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, margins: cellMargins, children: [new Paragraph({ children: [new TextRun("Outer radius (Ro)")] })] }),
            new TableCell({ borders, margins: cellMargins, children: [new Paragraph({ children: [new TextRun("1.56 m")] })] }),
            new TableCell({ borders, margins: cellMargins, children: [new Paragraph({ children: [new TextRun({ text: "0.052 m", bold: true })] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, margins: cellMargins, children: [new Paragraph({ children: [new TextRun("Inner radius (Ri)")] })] }),
            new TableCell({ borders, margins: cellMargins, children: [new Paragraph({ children: [new TextRun("1.44 m")] })] }),
            new TableCell({ borders, margins: cellMargins, children: [new Paragraph({ children: [new TextRun({ text: "0.048 m", bold: true })] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, margins: cellMargins, children: [new Paragraph({ children: [new TextRun("Height (h/D=1.5)")] })] }),
            new TableCell({ borders, margins: cellMargins, children: [new Paragraph({ children: [new TextRun("4.5 m")] })] }),
            new TableCell({ borders, margins: cellMargins, children: [new Paragraph({ children: [new TextRun({ text: "0.15 m", bold: true })] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, margins: cellMargins, children: [new Paragraph({ children: [new TextRun("Material properties")] })] }),
            new TableCell({ borders, margins: cellMargins, children: [new Paragraph({ children: [new TextRun("Original")] })] }),
            new TableCell({ borders, margins: cellMargins, children: [new Paragraph({ children: [new TextRun({ text: "UNCHANGED", bold: true })] })] })
          ]})
        ]
      }),
      
      new Paragraph({ spacing: { before: 300 }, heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "1.3 Critical Corrections", color: "E74C3C" })] }),
      
      new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Correction 1 - Soil Partitioning: ", bold: true }), new TextRun("Create a CAVITY in the soil where the anchor sits. Soil cannot occupy the same space as the anchor wall.")] }),
      new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Correction 2 - Mesh Gradation: ", bold: true }), new TextRun("Use fine mesh near anchor (0.01m) and coarse mesh at boundaries (0.05m) to stay under 1000 nodes.")] }),
      new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Correction 3 - Contact Settings: ", bold: true }), new TextRun("Use softened (exponential) contact instead of hard contact for convergence.")] }),
      
      new Paragraph({ spacing: { before: 200 } }),
      createNoteBox("Meshing is done AFTER all other setup (Parts, Materials, Assembly, Steps, Interactions, BCs, Loads). This is the correct sequence.", "important"),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // PART 2: CREATE MODEL
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("PART 2: Create New Model Database")] }),
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Step 1: Launch Abaqus and Create Model")] }),
      
      new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "1.1 Open Abaqus/CAE", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Launch Abaqus CAE from your Start menu")] }),
      
      new Paragraph({ spacing: { before: 200, after: 100 }, children: [new TextRun({ text: "1.2 Create New Model Database", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Click: "), new TextRun({ text: "File > New Model Database", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Select: "), new TextRun({ text: "With Standard/Explicit Model", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Click "), new TextRun({ text: "OK", bold: true })] }),
      
      new Paragraph({ spacing: { before: 200, after: 100 }, children: [new TextRun({ text: "1.3 Save Your Model", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Click: "), new TextRun({ text: "File > Save As", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("File name: "), new TextRun({ text: "DERA_Pullout_ze10D_h15D", bold: true })] }),
      
      new Paragraph({ spacing: { before: 200 } }),
      createNoteBox("Save frequently (Ctrl+S) every 10-15 minutes. Abaqus can crash unexpectedly.", "tip"),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // PART 3: CREATE SOIL
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("PART 3: Create Soil Domain (with Cavity)")] }),
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Step 2: Create Soil Part")] }),
      
      new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "2.1 Switch to Part Module", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Module dropdown (top-left) > "), new TextRun({ text: "Part", bold: true })] }),
      
      new Paragraph({ spacing: { before: 200, after: 100 }, children: [new TextRun({ text: "2.2 Create New Part", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Click: "), new TextRun({ text: "Part > Create", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Name: "), new TextRun({ text: "Soil", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Modeling Space: "), new TextRun({ text: "Axisymmetric", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Type: "), new TextRun({ text: "Deformable", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Base Feature: "), new TextRun({ text: "Shell", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Approximate size: "), new TextRun({ text: "5", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Click "), new TextRun({ text: "Continue", bold: true })] }),
      
      new Paragraph({ spacing: { before: 200, after: 100 }, children: [new TextRun({ text: "2.3 Draw Soil Domain Rectangle", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("In Sketcher, click: "), new TextRun({ text: "Create Lines > Rectangle", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("First corner: "), new TextRun({ text: "0, 0", bold: true }), new TextRun(" (press Enter)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Opposite corner: "), new TextRun({ text: "0.6, 1.5", bold: true }), new TextRun(" (press Enter)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Press "), new TextRun({ text: "Esc", bold: true }), new TextRun(" then click "), new TextRun({ text: "Done", bold: true })] }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // STEP 3: PARTITION FOR CAVITY
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "Step 3: Partition Soil for Anchor Cavity (CRITICAL)", color: "E74C3C" })] }),
      
      createNoteBox("This step is CRITICAL! Without this partition, soil occupies the anchor space, causing 'excessive element distortion' errors and model failure.", "warning"),
      
      new Paragraph({ spacing: { before: 200, after: 100 }, children: [new TextRun({ text: "3.1 Start Face Partition", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Click: "), new TextRun({ text: "Tools > Partition > Face > Sketch", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Click on the soil rectangle to select it")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Click "), new TextRun({ text: "Done", bold: true })] }),
      
      new Paragraph({ spacing: { before: 200, after: 100 }, children: [new TextRun({ text: "3.2 Draw Anchor Outer Boundary", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Click: "), new TextRun({ text: "Create Lines > Rectangle", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("First corner: "), new TextRun({ text: "0, 0.5", bold: true }), new TextRun(" (anchor bottom)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Opposite corner: "), new TextRun({ text: "0.052, 0.65", bold: true }), new TextRun(" (outer radius, top)")] }),
      
      new Paragraph({ spacing: { before: 200, after: 100 }, children: [new TextRun({ text: "3.3 Draw Anchor Inner Boundary", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("First corner: "), new TextRun({ text: "0, 0.5", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Opposite corner: "), new TextRun({ text: "0.048, 0.65", bold: true }), new TextRun(" (inner radius, top)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Press "), new TextRun({ text: "Esc", bold: true }), new TextRun(" then click "), new TextRun({ text: "Done", bold: true }), new TextRun(" twice")] }),
      
      new Paragraph({ spacing: { before: 200 } }),
      createNoteBox("The region between r=0.048 and r=0.052 (anchor wall zone) will NOT receive soil material. This creates the cavity.", "tip"),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // STEP 4: PARTITION FOR MESH
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Step 4: Partition Soil for Mesh Gradation")] }),
      
      new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "4.1 Create Mesh Zone Partition Lines", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Click: "), new TextRun({ text: "Tools > Partition > Face > Sketch", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Select soil face > "), new TextRun({ text: "Done", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Click: "), new TextRun({ text: "Create Lines > Line", bold: true })] }),
      
      new Paragraph({ spacing: { before: 100, after: 100 }, children: [new TextRun("Draw these partition lines:")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Vertical line at r = 0.15: from "), new TextRun({ text: "(0.15, 0)", bold: true }), new TextRun(" to "), new TextRun({ text: "(0.15, 1.5)", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Vertical line at r = 0.3: from "), new TextRun({ text: "(0.3, 0)", bold: true }), new TextRun(" to "), new TextRun({ text: "(0.3, 1.5)", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Horizontal line at y = 0.35: from "), new TextRun({ text: "(0, 0.35)", bold: true }), new TextRun(" to "), new TextRun({ text: "(0.6, 0.35)", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Horizontal line at y = 0.8: from "), new TextRun({ text: "(0, 0.8)", bold: true }), new TextRun(" to "), new TextRun({ text: "(0.6, 0.8)", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Press "), new TextRun({ text: "Esc", bold: true }), new TextRun(" then "), new TextRun({ text: "Done", bold: true }), new TextRun(" twice")] }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // PART 4: CREATE ANCHOR
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("PART 4: Create DERA Anchor")] }),
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Step 5: Create Anchor Part")] }),
      
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Click: "), new TextRun({ text: "Part > Create", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Name: "), new TextRun({ text: "DERA_Anchor", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Modeling Space: "), new TextRun({ text: "Axisymmetric", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Type: "), new TextRun({ text: "Deformable", bold: true }), new TextRun(", Base Feature: "), new TextRun({ text: "Shell", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Approximate size: "), new TextRun({ text: "0.3", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Click "), new TextRun({ text: "Continue", bold: true })] }),
      
      new Paragraph({ spacing: { before: 200, after: 100 }, children: [new TextRun({ text: "Draw Anchor Wall (Connected Lines):", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Point 1: "), new TextRun({ text: "0.048, 0", bold: true }), new TextRun(" (bottom inner)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Point 2: "), new TextRun({ text: "0.052, 0", bold: true }), new TextRun(" (bottom outer)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Point 3: "), new TextRun({ text: "0.052, 0.15", bold: true }), new TextRun(" (top outer)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Point 4: "), new TextRun({ text: "0.048, 0.15", bold: true }), new TextRun(" (top inner)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Point 5: "), new TextRun({ text: "0.048, 0", bold: true }), new TextRun(" (close rectangle)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Press "), new TextRun({ text: "Esc > Done", bold: true })] }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // PART 5: MATERIALS
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("PART 5: Define Materials")] }),
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Step 6: Create Sand Material")] }),
      
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Module dropdown > "), new TextRun({ text: "Property", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Click: "), new TextRun({ text: "Material > Create", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Name: "), new TextRun({ text: "Sand_Medium_Dense", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("General > Density: "), new TextRun({ text: "2600", bold: true }), new TextRun(" kg/m³")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Mechanical > Elasticity > Elastic:")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, indent: { left: 720 }, children: [new TextRun("Young's Modulus: "), new TextRun({ text: "100000", bold: true }), new TextRun(" kPa")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, indent: { left: 720 }, children: [new TextRun("Poisson's Ratio: "), new TextRun({ text: "0.3", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Mechanical > Plasticity > Mohr Coulomb Plasticity:")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, indent: { left: 720 }, children: [new TextRun("Friction Angle: "), new TextRun({ text: "35", bold: true }), new TextRun(" degrees")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, indent: { left: 720 }, children: [new TextRun("Dilation Angle: "), new TextRun({ text: "7.5", bold: true }), new TextRun(" degrees")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, indent: { left: 720 }, children: [new TextRun("Mohr Coulomb Hardening tab: Cohesion = "), new TextRun({ text: "0.1", bold: true }), new TextRun(" kPa, Strain = "), new TextRun({ text: "0", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Click "), new TextRun({ text: "OK", bold: true })] }),
      
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Step 7: Create Steel Material")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Material > Create, Name: "), new TextRun({ text: "Steel_DERA", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Density: "), new TextRun({ text: "7850", bold: true }), new TextRun(" kg/m³")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Elastic: E = "), new TextRun({ text: "200000000", bold: true }), new TextRun(" kPa, ν = "), new TextRun({ text: "0.3", bold: true })] }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // PART 6: SECTIONS
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("PART 6: Create and Assign Sections")] }),
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Step 8: Create Soil Section")] }),
      
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Section > Create, Name: "), new TextRun({ text: "Soil_Section", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Category: "), new TextRun({ text: "Solid", bold: true }), new TextRun(", Type: "), new TextRun({ text: "Homogeneous", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Material: "), new TextRun({ text: "Sand_Medium_Dense", bold: true })] }),
      
      new Paragraph({ spacing: { before: 200, after: 100 }, children: [new TextRun({ text: "Assign to Soil (EXCLUDING anchor wall zone):", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("In model tree: Parts > Soil > Assign Section")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Hold Shift and select ALL regions EXCEPT the thin strip at r = 0.048 to 0.052")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Click Done, select Soil_Section, OK")] }),
      
      createNoteBox("DO NOT assign material to the region between r=0.048 and r=0.052. This is where the anchor wall goes - leave it empty.", "warning"),
      
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Step 9: Create Anchor Section")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Section > Create, Name: "), new TextRun({ text: "DERA_Section", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Material: "), new TextRun({ text: "Steel_DERA", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Assign to entire DERA_Anchor part")] }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // PART 7: ASSEMBLY
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("PART 7: Assembly")] }),
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Step 10: Assemble Parts")] }),
      
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Module dropdown > "), new TextRun({ text: "Assembly", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Instance > Create, select "), new TextRun({ text: "Soil", bold: true }), new TextRun(", Type: "), new TextRun({ text: "Independent", bold: true }), new TextRun(", OK")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Instance > Create, select "), new TextRun({ text: "DERA_Anchor", bold: true }), new TextRun(", Type: "), new TextRun({ text: "Independent", bold: true }), new TextRun(", OK")] }),
      
      new Paragraph({ spacing: { before: 200, after: 100 }, children: [new TextRun({ text: "Position Anchor:", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Instance > Translate, select DERA_Anchor-1")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Start point: "), new TextRun({ text: "0, 0", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("End point: "), new TextRun({ text: "0, 0.5", bold: true }), new TextRun(" (moves anchor to embedment depth)")] }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // PART 8: STEPS
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("PART 8: Define Analysis Steps")] }),
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Step 11: Create Geostatic Step")] }),
      
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Module > "), new TextRun({ text: "Step", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Step > Create, Name: "), new TextRun({ text: "Geostatic", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Procedure: "), new TextRun({ text: "General > Geostatic", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Basic tab: Nlgeom = "), new TextRun({ text: "ON", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Incrementation: Initial = "), new TextRun({ text: "0.01", bold: true }), new TextRun(", Min = "), new TextRun({ text: "1e-08", bold: true }), new TextRun(", Max = "), new TextRun({ text: "0.1", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Max increments: "), new TextRun({ text: "1000", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Other tab: Matrix storage = "), new TextRun({ text: "Unsymmetric", bold: true })] }),
      
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Step 12: Create Pullout Step")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Step > Create, Name: "), new TextRun({ text: "Pullout", bold: true }), new TextRun(", after Geostatic")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Procedure: "), new TextRun({ text: "General > Static, General", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Same settings: Nlgeom ON, Initial 0.01, Min 1e-08, Max 0.1, Unsymmetric")] }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // PART 9: INTERACTIONS
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("PART 9: Define Interactions (Contact)")] }),
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Step 13: Create Contact Property (CORRECTED)")] }),
      
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Module > "), new TextRun({ text: "Interaction", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Interaction > Property > Create, Name: "), new TextRun({ text: "DERA_Soil_Contact", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Mechanical > Tangential: Penalty, Friction = "), new TextRun({ text: "0.6", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Mechanical > Normal: "), new TextRun({ text: "Exponential (NOT Hard!)", bold: true, color: "E74C3C" })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, indent: { left: 720 }, children: [new TextRun("Pressure at zero clearance: "), new TextRun({ text: "100000", bold: true }), new TextRun(" kPa")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, indent: { left: 720 }, children: [new TextRun("Clearance at zero pressure: "), new TextRun({ text: "0.0001", bold: true }), new TextRun(" m")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Allow separation: "), new TextRun({ text: "Yes", bold: true })] }),
      
      createNoteBox("Using exponential (softened) contact instead of hard contact is essential for convergence in scaled models.", "important"),
      
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Step 14: Create Surfaces and Contact Pairs")] }),
      new Paragraph({ children: [new TextRun("Create surfaces (Tools > Surface > Create):")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "Anchor_Outer_Surface", bold: true }), new TextRun(": outer edge of anchor (r = 0.052)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "Anchor_Inner_Surface", bold: true }), new TextRun(": inner edge of anchor (r = 0.048)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "Soil_Outer_Contact", bold: true }), new TextRun(": soil edge at r = 0.052")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "Soil_Plug_Surface", bold: true }), new TextRun(": soil edge at r = 0.048")] }),
      
      new Paragraph({ spacing: { before: 100 }, children: [new TextRun("Create contact interactions (Interaction > Create, Step: Initial, Surface-to-surface):")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "Contact_Outer", bold: true }), new TextRun(": Master = Soil_Outer_Contact, Slave = Anchor_Outer_Surface")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "Contact_Inner", bold: true }), new TextRun(": Master = Soil_Plug_Surface, Slave = Anchor_Inner_Surface")] }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // PART 10: BOUNDARY CONDITIONS
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("PART 10: Apply Boundary Conditions")] }),
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Step 15: Create Boundary Conditions")] }),
      
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Module > "), new TextRun({ text: "Load", bold: true })] }),
      
      new Paragraph({ spacing: { before: 100 }, children: [new TextRun({ text: "BC_Bottom_Fixed (Geostatic step):", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Type: Displacement/Rotation, select bottom edge (y = 0)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("U1 = 0, U2 = 0")] }),
      
      new Paragraph({ spacing: { before: 100 }, children: [new TextRun({ text: "BC_Symmetry (Geostatic step):", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Type: Symmetry/Antisymmetry/Encastre, select left edge (r = 0)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Select: "), new TextRun({ text: "XSYMM", bold: true })] }),
      
      new Paragraph({ spacing: { before: 100 }, children: [new TextRun({ text: "BC_Lateral (Geostatic step):", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Type: Displacement/Rotation, select right edge (r = 0.6)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("U1 = 0 only")] }),
      
      new Paragraph({ spacing: { before: 100 }, children: [new TextRun({ text: "BC_Anchor_Pullout (Pullout step):", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Type: Displacement/Rotation, select top edge of anchor")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("U2 = "), new TextRun({ text: "0.05", bold: true }), new TextRun(" (50 mm upward)")] }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // PART 11: LOADS
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("PART 11: Apply Loads")] }),
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Step 16: Apply Gravity")] }),
      
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Load > Create, Name: "), new TextRun({ text: "Gravity_Load", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Step: "), new TextRun({ text: "Geostatic", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Type: "), new TextRun({ text: "Gravity", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Select all soil regions")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Component 2: "), new TextRun({ text: "-9.81", bold: true }), new TextRun(" m/s²")] }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // PART 12: MESH
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("PART 12: Mesh Generation (FINAL STEP BEFORE JOB)")] }),
      
      createNoteBox("Mesh AFTER all other setup is complete. This ensures mesh is applied to final geometry with all partitions.", "important"),
      
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Step 17: Mesh the Soil")] }),
      
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Module > "), new TextRun({ text: "Mesh", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Select part: "), new TextRun({ text: "Soil-1", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Mesh > Controls: Quad, Structured")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Mesh > Element Type: Axisymmetric Stress, "), new TextRun({ text: "CAX4R", bold: true })] }),
      
      new Paragraph({ spacing: { before: 100 }, children: [new TextRun({ text: "Seed edges with gradation:", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Fine zone (r < 0.15m): Seed > Edge, size = "), new TextRun({ text: "0.01", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Medium zone (0.15 < r < 0.3): size = "), new TextRun({ text: "0.03", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Coarse zone (r > 0.3): size = "), new TextRun({ text: "0.05", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Mesh > Part > Yes")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Verify: Query > Mesh (should be under 900 nodes)")] }),
      
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Step 18: Mesh the Anchor")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Select part: "), new TextRun({ text: "DERA_Anchor-1", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Element Type: "), new TextRun({ text: "CAX4R", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Seed > Part, size = "), new TextRun({ text: "0.005", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Mesh > Part > Yes")] }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // PART 13: JOB
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("PART 13: Create and Run Job")] }),
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Step 19: Submit Job")] }),
      
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Module > "), new TextRun({ text: "Job", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Job > Create, Name: "), new TextRun({ text: "DERA_Pullout_ze10D_h15D", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Job > Submit")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Job > Monitor (watch for "), new TextRun({ text: "Completed", bold: true, color: "27AE60" }), new TextRun(")")] }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // PART 14: RESULTS
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("PART 14: View Results")] }),
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Step 20: Post-Processing")] }),
      
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Job > Results (opens Visualization)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Plot > Contours > On Deformed Shape")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Field output: U (displacement), U2 for vertical")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Field output: S (stress), S22 or Mises")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Tools > XY Data > Create > ODB history output > RF2")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Peak RF2 = ultimate pullout capacity")] }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // APPENDIX
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("APPENDIX: Troubleshooting")] }),
      
      new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "\"TOO MANY ATTEMPTS MADE FOR THIS INCREMENT\"", bold: true, color: "E74C3C" })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Use softened contact (exponential), not hard contact")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Reduce initial increment to 0.001")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Check soil cavity exists where anchor is placed")] }),
      
      new Paragraph({ spacing: { before: 200, after: 100 }, children: [new TextRun({ text: "\"EXCESSIVE ELEMENT DISTORTION\"", bold: true, color: "E74C3C" })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Ensure soil does NOT fill anchor wall space")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Refine mesh near anchor")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Enable NLGEOM")] }),
      
      new Paragraph({ spacing: { before: 200, after: 100 }, children: [new TextRun({ text: "\"NODE COUNT EXCEEDS LIMIT\"", bold: true, color: "E74C3C" })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Increase mesh seed sizes")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("Use mesh gradation")] }),
      
      new Paragraph({
        spacing: { before: 400 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "— End of Guide —", italics: true, color: "888888" })]
      })
    ]
  }]
});

// Write the document
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("DERA_Tutorial_Corrected_Guide.docx", buffer);
  console.log("Document created successfully!");
});