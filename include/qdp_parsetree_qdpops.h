// FnAdjoint
template <>
struct OpVisitor<FnAdjoint, ParseTag> : public BracketPrinter<FnAdjoint>
{ 
  static bool visit(FnAdjoint op, const ParseTag & t) 
    { 
      t.ossCode << "FnAdjoint()";
      return true;
    }
};

// FnConjugate
template <>
struct OpVisitor<FnConjugate, ParseTag> : public BracketPrinter<FnConjugate>
{ 
  static bool visit(FnConjugate op, const ParseTag & t) 
    { 
      t.ossCode << "FnConjugate()";
      return true;
    }
};

// FnTranspose
template <>
struct OpVisitor<FnTranspose, ParseTag> : public BracketPrinter<FnTranspose>
{ 
  static bool visit(FnTranspose op, const ParseTag & t) 
    { 
      t.ossCode << "FnTranspose()";
      return true;
    }
};

// FnTransposeColor
template <>
struct OpVisitor<FnTransposeColor, ParseTag> : public BracketPrinter<FnTransposeColor>
{ 
  static bool visit(FnTransposeColor op, const ParseTag & t) 
    { 
      t.ossCode << "FnTransposeColor()";
      return true;
    }
};

// FnTransposeSpin
template <>
struct OpVisitor<FnTransposeSpin, ParseTag> : public BracketPrinter<FnTransposeSpin>
{ 
  static bool visit(FnTransposeSpin op, const ParseTag & t) 
    { 
      t.ossCode << "FnTransposeSpin()";
      return true;
    }
};

// FnTrace
template <>
struct OpVisitor<FnTrace, ParseTag> : public BracketPrinter<FnTrace>
{ 
  static bool visit(FnTrace op, const ParseTag & t) 
    { 
      t.ossCode << "FnTrace()";
      return true;
    }
};

// FnRealTrace
template <>
struct OpVisitor<FnRealTrace, ParseTag> : public BracketPrinter<FnRealTrace>
{ 
  static bool visit(FnRealTrace op, const ParseTag & t) 
    { 
      t.ossCode << "FnRealTrace()";
      return true;
    }
};

// FnImagTrace
template <>
struct OpVisitor<FnImagTrace, ParseTag> : public BracketPrinter<FnImagTrace>
{ 
  static bool visit(FnImagTrace op, const ParseTag & t) 
    { 
      t.ossCode << "FnImagTrace()";
      return true;
    }
};

// FnTraceColor
template <>
struct OpVisitor<FnTraceColor, ParseTag> : public BracketPrinter<FnTraceColor>
{ 
  static bool visit(FnTraceColor op, const ParseTag & t) 
    { 
      t.ossCode << "FnTraceColor()";
      return true;
    }
};

// FnTraceSpin
template <>
struct OpVisitor<FnTraceSpin, ParseTag> : public BracketPrinter<FnTraceSpin>
{ 
  static bool visit(FnTraceSpin op, const ParseTag & t) 
    { 
      t.ossCode << "FnTraceSpin()";
      return true;
    }
};

// FnReal
template <>
struct OpVisitor<FnReal, ParseTag> : public BracketPrinter<FnReal>
{ 
  static bool visit(FnReal op, const ParseTag & t) 
    { 
      t.ossCode << "FnReal()";
      return true;
    }
};

// FnImag
template <>
struct OpVisitor<FnImag, ParseTag> : public BracketPrinter<FnImag>
{ 
  static bool visit(FnImag op, const ParseTag & t) 
    { 
      t.ossCode << "FnImag()";
      return true;
    }
};

// FnLocalNorm2
template <>
struct OpVisitor<FnLocalNorm2, ParseTag> : public BracketPrinter<FnLocalNorm2>
{ 
  static bool visit(FnLocalNorm2 op, const ParseTag & t) 
    { 
      t.ossCode << "FnLocalNorm2()";
      return true;
    }
};

// FnTimesI
template <>
struct OpVisitor<FnTimesI, ParseTag> : public BracketPrinter<FnTimesI>
{ 
  static bool visit(FnTimesI op, const ParseTag & t) 
    { 
      t.ossCode << "FnTimesI()";
      return true;
    }
};

// FnTimesMinusI
template <>
struct OpVisitor<FnTimesMinusI, ParseTag> : public BracketPrinter<FnTimesMinusI>
{ 
  static bool visit(FnTimesMinusI op, const ParseTag & t) 
    { 
      t.ossCode << "FnTimesMinusI()";
      return true;
    }
};

// FnSeedToFloat
template <>
struct OpVisitor<FnSeedToFloat, ParseTag> : public BracketPrinter<FnSeedToFloat>
{ 
  static bool visit(FnSeedToFloat op, const ParseTag & t) 
    { 
      t.ossCode << "FnSeedToFloat()";
      return true;
    }
};

// FnSpinProjectDir0Plus
template <>
struct OpVisitor<FnSpinProjectDir0Plus, ParseTag> : public BracketPrinter<FnSpinProjectDir0Plus>
{ 
  static bool visit(FnSpinProjectDir0Plus op, const ParseTag & t) 
    { 
      t.ossCode << "FnSpinProjectDir0Plus()";
      return true;
    }
};

// FnSpinProjectDir1Plus
template <>
struct OpVisitor<FnSpinProjectDir1Plus, ParseTag> : public BracketPrinter<FnSpinProjectDir1Plus>
{ 
  static bool visit(FnSpinProjectDir1Plus op, const ParseTag & t) 
    { 
      t.ossCode << "FnSpinProjectDir1Plus()";
      return true;
    }
};

// FnSpinProjectDir2Plus
template <>
struct OpVisitor<FnSpinProjectDir2Plus, ParseTag> : public BracketPrinter<FnSpinProjectDir2Plus>
{ 
  static bool visit(FnSpinProjectDir2Plus op, const ParseTag & t) 
    { 
      t.ossCode << "FnSpinProjectDir2Plus()";
      return true;
    }
};

// FnSpinProjectDir3Plus
template <>
struct OpVisitor<FnSpinProjectDir3Plus, ParseTag> : public BracketPrinter<FnSpinProjectDir3Plus>
{ 
  static bool visit(FnSpinProjectDir3Plus op, const ParseTag & t) 
    { 
      t.ossCode << "FnSpinProjectDir3Plus()";
      return true;
    }
};

// FnSpinProjectDir0Minus
template <>
struct OpVisitor<FnSpinProjectDir0Minus, ParseTag> : public BracketPrinter<FnSpinProjectDir0Minus>
{ 
  static bool visit(FnSpinProjectDir0Minus op, const ParseTag & t) 
    { 
      t.ossCode << "FnSpinProjectDir0Minus()";
      return true;
    }
};

// FnSpinProjectDir1Minus
template <>
struct OpVisitor<FnSpinProjectDir1Minus, ParseTag> : public BracketPrinter<FnSpinProjectDir1Minus>
{ 
  static bool visit(FnSpinProjectDir1Minus op, const ParseTag & t) 
    { 
      t.ossCode << "FnSpinProjectDir1Minus()";
      return true;
    }
};

// FnSpinProjectDir2Minus
template <>
struct OpVisitor<FnSpinProjectDir2Minus, ParseTag> : public BracketPrinter<FnSpinProjectDir2Minus>
{ 
  static bool visit(FnSpinProjectDir2Minus op, const ParseTag & t) 
    { 
      t.ossCode << "FnSpinProjectDir2Minus()";
      return true;
    }
};

// FnSpinProjectDir3Minus
template <>
struct OpVisitor<FnSpinProjectDir3Minus, ParseTag> : public BracketPrinter<FnSpinProjectDir3Minus>
{ 
  static bool visit(FnSpinProjectDir3Minus op, const ParseTag & t) 
    { 
      t.ossCode << "FnSpinProjectDir3Minus()";
      return true;
    }
};

// FnSpinReconstructDir0Plus
template <>
struct OpVisitor<FnSpinReconstructDir0Plus, ParseTag> : public BracketPrinter<FnSpinReconstructDir0Plus>
{ 
  static bool visit(FnSpinReconstructDir0Plus op, const ParseTag & t) 
    { 
      t.ossCode << "FnSpinReconstructDir0Plus()";
      return true;
    }
};

// FnSpinReconstructDir1Plus
template <>
struct OpVisitor<FnSpinReconstructDir1Plus, ParseTag> : public BracketPrinter<FnSpinReconstructDir1Plus>
{ 
  static bool visit(FnSpinReconstructDir1Plus op, const ParseTag & t) 
    { 
      t.ossCode << "FnSpinReconstructDir1Plus()";
      return true;
    }
};

// FnSpinReconstructDir2Plus
template <>
struct OpVisitor<FnSpinReconstructDir2Plus, ParseTag> : public BracketPrinter<FnSpinReconstructDir2Plus>
{ 
  static bool visit(FnSpinReconstructDir2Plus op, const ParseTag & t) 
    { 
      t.ossCode << "FnSpinReconstructDir2Plus()";
      return true;
    }
};

// FnSpinReconstructDir3Plus
template <>
struct OpVisitor<FnSpinReconstructDir3Plus, ParseTag> : public BracketPrinter<FnSpinReconstructDir3Plus>
{ 
  static bool visit(FnSpinReconstructDir3Plus op, const ParseTag & t) 
    { 
      t.ossCode << "FnSpinReconstructDir3Plus()";
      return true;
    }
};

// FnSpinReconstructDir0Minus
template <>
struct OpVisitor<FnSpinReconstructDir0Minus, ParseTag> : public BracketPrinter<FnSpinReconstructDir0Minus>
{ 
  static bool visit(FnSpinReconstructDir0Minus op, const ParseTag & t) 
    { 
      t.ossCode << "FnSpinReconstructDir0Minus()";
      return true;
    }
};

// FnSpinReconstructDir1Minus
template <>
struct OpVisitor<FnSpinReconstructDir1Minus, ParseTag> : public BracketPrinter<FnSpinReconstructDir1Minus>
{ 
  static bool visit(FnSpinReconstructDir1Minus op, const ParseTag & t) 
    { 
      t.ossCode << "FnSpinReconstructDir1Minus()";
      return true;
    }
};

// FnSpinReconstructDir2Minus
template <>
struct OpVisitor<FnSpinReconstructDir2Minus, ParseTag> : public BracketPrinter<FnSpinReconstructDir2Minus>
{ 
  static bool visit(FnSpinReconstructDir2Minus op, const ParseTag & t) 
    { 
      t.ossCode << "FnSpinReconstructDir2Minus()";
      return true;
    }
};

// FnSpinReconstructDir3Minus
template <>
struct OpVisitor<FnSpinReconstructDir3Minus, ParseTag> : public BracketPrinter<FnSpinReconstructDir3Minus>
{ 
  static bool visit(FnSpinReconstructDir3Minus op, const ParseTag & t) 
    { 
      t.ossCode << "FnSpinReconstructDir3Minus()";
      return true;
    }
};

// FnChiralProjectPlus
template <>
struct OpVisitor<FnChiralProjectPlus, ParseTag> : public BracketPrinter<FnChiralProjectPlus>
{ 
  static bool visit(FnChiralProjectPlus op, const ParseTag & t) 
    { 
      t.ossCode << "FnChiralProjectPlus()";
      return true;
    }
};

// FnChiralProjectMinus
template <>
struct OpVisitor<FnChiralProjectMinus, ParseTag> : public BracketPrinter<FnChiralProjectMinus>
{ 
  static bool visit(FnChiralProjectMinus op, const ParseTag & t) 
    { 
      t.ossCode << "FnChiralProjectMinus()";
      return true;
    }
};

// FnCmplx
template <>
struct OpVisitor<FnCmplx, ParseTag> : public BracketPrinter<FnCmplx>
{ 
  static bool visit(FnCmplx op, const ParseTag & t) 
    { 
      t.ossCode << "FnCmplx()";
      return true;
    }
};

// FnOuterProduct
template <>
struct OpVisitor<FnOuterProduct, ParseTag> : public BracketPrinter<FnOuterProduct>
{ 
  static bool visit(FnOuterProduct op, const ParseTag & t) 
    { 
      t.ossCode << "FnOuterProduct()";
      return true;
    }
};

// FnColorVectorContract
template <>
struct OpVisitor<FnColorVectorContract, ParseTag> : public BracketPrinter<FnColorVectorContract>
{ 
  static bool visit(FnColorVectorContract op, const ParseTag & t) 
    { 
      t.ossCode << "FnColorVectorContract()";
      return true;
    }
};

// FnColorCrossProduct
template <>
struct OpVisitor<FnColorCrossProduct, ParseTag> : public BracketPrinter<FnColorCrossProduct>
{ 
  static bool visit(FnColorCrossProduct op, const ParseTag & t) 
    { 
      t.ossCode << "FnColorCrossProduct()";
      return true;
    }
};

// FnLocalInnerProduct
template <>
struct OpVisitor<FnLocalInnerProduct, ParseTag> : public BracketPrinter<FnLocalInnerProduct>
{ 
  static bool visit(FnLocalInnerProduct op, const ParseTag & t) 
    { 
      t.ossCode << "FnLocalInnerProduct()";
      return true;
    }
};

// FnLocalInnerProductReal
template <>
struct OpVisitor<FnLocalInnerProductReal, ParseTag> : public BracketPrinter<FnLocalInnerProductReal>
{ 
  static bool visit(FnLocalInnerProductReal op, const ParseTag & t) 
    { 
      t.ossCode << "FnLocalInnerProductReal()";
      return true;
    }
};

// FnQuarkContract13
template <>
struct OpVisitor<FnQuarkContract13, ParseTag> : public BracketPrinter<FnQuarkContract13>
{ 
  static bool visit(FnQuarkContract13 op, const ParseTag & t) 
    { 
      t.ossCode << "FnQuarkContract13()";
      return true;
    }
};

// FnQuarkContract14
template <>
struct OpVisitor<FnQuarkContract14, ParseTag> : public BracketPrinter<FnQuarkContract14>
{ 
  static bool visit(FnQuarkContract14 op, const ParseTag & t) 
    { 
      t.ossCode << "FnQuarkContract14()";
      return true;
    }
};

// FnQuarkContract23
template <>
struct OpVisitor<FnQuarkContract23, ParseTag> : public BracketPrinter<FnQuarkContract23>
{ 
  static bool visit(FnQuarkContract23 op, const ParseTag & t) 
    { 
      t.ossCode << "FnQuarkContract23()";
      return true;
    }
};

// FnQuarkContract24
template <>
struct OpVisitor<FnQuarkContract24, ParseTag> : public BracketPrinter<FnQuarkContract24>
{ 
  static bool visit(FnQuarkContract24 op, const ParseTag & t) 
    { 
      t.ossCode << "FnQuarkContract24()";
      return true;
    }
};

// FnQuarkContract12
template <>
struct OpVisitor<FnQuarkContract12, ParseTag> : public BracketPrinter<FnQuarkContract12>
{ 
  static bool visit(FnQuarkContract12 op, const ParseTag & t) 
    { 
      t.ossCode << "FnQuarkContract12()";
      return true;
    }
};

// FnQuarkContract34
template <>
struct OpVisitor<FnQuarkContract34, ParseTag> : public BracketPrinter<FnQuarkContract34>
{ 
  static bool visit(FnQuarkContract34 op, const ParseTag & t) 
    { 
      t.ossCode << "FnQuarkContract34()";
      return true;
    }
};

// FnColorContract
template <>
struct OpVisitor<FnColorContract, ParseTag> : public BracketPrinter<FnColorContract>
{ 
  static bool visit(FnColorContract op, const ParseTag & t) 
    { 
      t.ossCode << "FnColorContract()";
      return true;
    }
};

