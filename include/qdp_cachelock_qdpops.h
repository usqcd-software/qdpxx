// FnAdjoint
template <>
struct OpVisitor<FnAdjoint, LockTag> : public BracketPrinter<FnAdjoint>
{ 
  static bool visit(FnAdjoint op, const LockTag & t) 
    { 
      // t.ossCode << "FnAdjoint()";
      return true;
    }
};

// FnConjugate
template <>
struct OpVisitor<FnConjugate, LockTag> : public BracketPrinter<FnConjugate>
{ 
  static bool visit(FnConjugate op, const LockTag & t) 
    { 
      // t.ossCode << "FnConjugate()";
      return true;
    }
};

// FnTranspose
template <>
struct OpVisitor<FnTranspose, LockTag> : public BracketPrinter<FnTranspose>
{ 
  static bool visit(FnTranspose op, const LockTag & t) 
    { 
      // t.ossCode << "FnTranspose()";
      return true;
    }
};

// FnTransposeColor
template <>
struct OpVisitor<FnTransposeColor, LockTag> : public BracketPrinter<FnTransposeColor>
{ 
  static bool visit(FnTransposeColor op, const LockTag & t) 
    { 
      // t.ossCode << "FnTransposeColor()";
      return true;
    }
};

// FnTransposeSpin
template <>
struct OpVisitor<FnTransposeSpin, LockTag> : public BracketPrinter<FnTransposeSpin>
{ 
  static bool visit(FnTransposeSpin op, const LockTag & t) 
    { 
      // t.ossCode << "FnTransposeSpin()";
      return true;
    }
};

// FnTrace
template <>
struct OpVisitor<FnTrace, LockTag> : public BracketPrinter<FnTrace>
{ 
  static bool visit(FnTrace op, const LockTag & t) 
    { 
      // t.ossCode << "FnTrace()";
      return true;
    }
};

// FnRealTrace
template <>
struct OpVisitor<FnRealTrace, LockTag> : public BracketPrinter<FnRealTrace>
{ 
  static bool visit(FnRealTrace op, const LockTag & t) 
    { 
      // t.ossCode << "FnRealTrace()";
      return true;
    }
};

// FnImagTrace
template <>
struct OpVisitor<FnImagTrace, LockTag> : public BracketPrinter<FnImagTrace>
{ 
  static bool visit(FnImagTrace op, const LockTag & t) 
    { 
      // t.ossCode << "FnImagTrace()";
      return true;
    }
};

// FnTraceColor
template <>
struct OpVisitor<FnTraceColor, LockTag> : public BracketPrinter<FnTraceColor>
{ 
  static bool visit(FnTraceColor op, const LockTag & t) 
    { 
      // t.ossCode << "FnTraceColor()";
      return true;
    }
};

// FnTraceSpin
template <>
struct OpVisitor<FnTraceSpin, LockTag> : public BracketPrinter<FnTraceSpin>
{ 
  static bool visit(FnTraceSpin op, const LockTag & t) 
    { 
      // t.ossCode << "FnTraceSpin()";
      return true;
    }
};

// FnReal
template <>
struct OpVisitor<FnReal, LockTag> : public BracketPrinter<FnReal>
{ 
  static bool visit(FnReal op, const LockTag & t) 
    { 
      // t.ossCode << "FnReal()";
      return true;
    }
};

// FnImag
template <>
struct OpVisitor<FnImag, LockTag> : public BracketPrinter<FnImag>
{ 
  static bool visit(FnImag op, const LockTag & t) 
    { 
      // t.ossCode << "FnImag()";
      return true;
    }
};

// FnLocalNorm2
template <>
struct OpVisitor<FnLocalNorm2, LockTag> : public BracketPrinter<FnLocalNorm2>
{ 
  static bool visit(FnLocalNorm2 op, const LockTag & t) 
    { 
      // t.ossCode << "FnLocalNorm2()";
      return true;
    }
};

// FnTimesI
template <>
struct OpVisitor<FnTimesI, LockTag> : public BracketPrinter<FnTimesI>
{ 
  static bool visit(FnTimesI op, const LockTag & t) 
    { 
      // t.ossCode << "FnTimesI()";
      return true;
    }
};

// FnTimesMinusI
template <>
struct OpVisitor<FnTimesMinusI, LockTag> : public BracketPrinter<FnTimesMinusI>
{ 
  static bool visit(FnTimesMinusI op, const LockTag & t) 
    { 
      // t.ossCode << "FnTimesMinusI()";
      return true;
    }
};

// FnSeedToFloat
template <>
struct OpVisitor<FnSeedToFloat, LockTag> : public BracketPrinter<FnSeedToFloat>
{ 
  static bool visit(FnSeedToFloat op, const LockTag & t) 
    { 
      // t.ossCode << "FnSeedToFloat()";
      return true;
    }
};

// FnSpinProjectDir0Plus
template <>
struct OpVisitor<FnSpinProjectDir0Plus, LockTag> : public BracketPrinter<FnSpinProjectDir0Plus>
{ 
  static bool visit(FnSpinProjectDir0Plus op, const LockTag & t) 
    { 
      // t.ossCode << "FnSpinProjectDir0Plus()";
      return true;
    }
};

// FnSpinProjectDir1Plus
template <>
struct OpVisitor<FnSpinProjectDir1Plus, LockTag> : public BracketPrinter<FnSpinProjectDir1Plus>
{ 
  static bool visit(FnSpinProjectDir1Plus op, const LockTag & t) 
    { 
      // t.ossCode << "FnSpinProjectDir1Plus()";
      return true;
    }
};

// FnSpinProjectDir2Plus
template <>
struct OpVisitor<FnSpinProjectDir2Plus, LockTag> : public BracketPrinter<FnSpinProjectDir2Plus>
{ 
  static bool visit(FnSpinProjectDir2Plus op, const LockTag & t) 
    { 
      // t.ossCode << "FnSpinProjectDir2Plus()";
      return true;
    }
};

// FnSpinProjectDir3Plus
template <>
struct OpVisitor<FnSpinProjectDir3Plus, LockTag> : public BracketPrinter<FnSpinProjectDir3Plus>
{ 
  static bool visit(FnSpinProjectDir3Plus op, const LockTag & t) 
    { 
      // t.ossCode << "FnSpinProjectDir3Plus()";
      return true;
    }
};

// FnSpinProjectDir0Minus
template <>
struct OpVisitor<FnSpinProjectDir0Minus, LockTag> : public BracketPrinter<FnSpinProjectDir0Minus>
{ 
  static bool visit(FnSpinProjectDir0Minus op, const LockTag & t) 
    { 
      // t.ossCode << "FnSpinProjectDir0Minus()";
      return true;
    }
};

// FnSpinProjectDir1Minus
template <>
struct OpVisitor<FnSpinProjectDir1Minus, LockTag> : public BracketPrinter<FnSpinProjectDir1Minus>
{ 
  static bool visit(FnSpinProjectDir1Minus op, const LockTag & t) 
    { 
      // t.ossCode << "FnSpinProjectDir1Minus()";
      return true;
    }
};

// FnSpinProjectDir2Minus
template <>
struct OpVisitor<FnSpinProjectDir2Minus, LockTag> : public BracketPrinter<FnSpinProjectDir2Minus>
{ 
  static bool visit(FnSpinProjectDir2Minus op, const LockTag & t) 
    { 
      // t.ossCode << "FnSpinProjectDir2Minus()";
      return true;
    }
};

// FnSpinProjectDir3Minus
template <>
struct OpVisitor<FnSpinProjectDir3Minus, LockTag> : public BracketPrinter<FnSpinProjectDir3Minus>
{ 
  static bool visit(FnSpinProjectDir3Minus op, const LockTag & t) 
    { 
      // t.ossCode << "FnSpinProjectDir3Minus()";
      return true;
    }
};

// FnSpinReconstructDir0Plus
template <>
struct OpVisitor<FnSpinReconstructDir0Plus, LockTag> : public BracketPrinter<FnSpinReconstructDir0Plus>
{ 
  static bool visit(FnSpinReconstructDir0Plus op, const LockTag & t) 
    { 
      // t.ossCode << "FnSpinReconstructDir0Plus()";
      return true;
    }
};

// FnSpinReconstructDir1Plus
template <>
struct OpVisitor<FnSpinReconstructDir1Plus, LockTag> : public BracketPrinter<FnSpinReconstructDir1Plus>
{ 
  static bool visit(FnSpinReconstructDir1Plus op, const LockTag & t) 
    { 
      // t.ossCode << "FnSpinReconstructDir1Plus()";
      return true;
    }
};

// FnSpinReconstructDir2Plus
template <>
struct OpVisitor<FnSpinReconstructDir2Plus, LockTag> : public BracketPrinter<FnSpinReconstructDir2Plus>
{ 
  static bool visit(FnSpinReconstructDir2Plus op, const LockTag & t) 
    { 
      // t.ossCode << "FnSpinReconstructDir2Plus()";
      return true;
    }
};

// FnSpinReconstructDir3Plus
template <>
struct OpVisitor<FnSpinReconstructDir3Plus, LockTag> : public BracketPrinter<FnSpinReconstructDir3Plus>
{ 
  static bool visit(FnSpinReconstructDir3Plus op, const LockTag & t) 
    { 
      // t.ossCode << "FnSpinReconstructDir3Plus()";
      return true;
    }
};

// FnSpinReconstructDir0Minus
template <>
struct OpVisitor<FnSpinReconstructDir0Minus, LockTag> : public BracketPrinter<FnSpinReconstructDir0Minus>
{ 
  static bool visit(FnSpinReconstructDir0Minus op, const LockTag & t) 
    { 
      // t.ossCode << "FnSpinReconstructDir0Minus()";
      return true;
    }
};

// FnSpinReconstructDir1Minus
template <>
struct OpVisitor<FnSpinReconstructDir1Minus, LockTag> : public BracketPrinter<FnSpinReconstructDir1Minus>
{ 
  static bool visit(FnSpinReconstructDir1Minus op, const LockTag & t) 
    { 
      // t.ossCode << "FnSpinReconstructDir1Minus()";
      return true;
    }
};

// FnSpinReconstructDir2Minus
template <>
struct OpVisitor<FnSpinReconstructDir2Minus, LockTag> : public BracketPrinter<FnSpinReconstructDir2Minus>
{ 
  static bool visit(FnSpinReconstructDir2Minus op, const LockTag & t) 
    { 
      // t.ossCode << "FnSpinReconstructDir2Minus()";
      return true;
    }
};

// FnSpinReconstructDir3Minus
template <>
struct OpVisitor<FnSpinReconstructDir3Minus, LockTag> : public BracketPrinter<FnSpinReconstructDir3Minus>
{ 
  static bool visit(FnSpinReconstructDir3Minus op, const LockTag & t) 
    { 
      // t.ossCode << "FnSpinReconstructDir3Minus()";
      return true;
    }
};

// FnChiralProjectPlus
template <>
struct OpVisitor<FnChiralProjectPlus, LockTag> : public BracketPrinter<FnChiralProjectPlus>
{ 
  static bool visit(FnChiralProjectPlus op, const LockTag & t) 
    { 
      // t.ossCode << "FnChiralProjectPlus()";
      return true;
    }
};

// FnChiralProjectMinus
template <>
struct OpVisitor<FnChiralProjectMinus, LockTag> : public BracketPrinter<FnChiralProjectMinus>
{ 
  static bool visit(FnChiralProjectMinus op, const LockTag & t) 
    { 
      // t.ossCode << "FnChiralProjectMinus()";
      return true;
    }
};

// FnCmplx
template <>
struct OpVisitor<FnCmplx, LockTag> : public BracketPrinter<FnCmplx>
{ 
  static bool visit(FnCmplx op, const LockTag & t) 
    { 
      // t.ossCode << "FnCmplx()";
      return true;
    }
};

// FnOuterProduct
template <>
struct OpVisitor<FnOuterProduct, LockTag> : public BracketPrinter<FnOuterProduct>
{ 
  static bool visit(FnOuterProduct op, const LockTag & t) 
    { 
      // t.ossCode << "FnOuterProduct()";
      return true;
    }
};

// FnColorVectorContract
template <>
struct OpVisitor<FnColorVectorContract, LockTag> : public BracketPrinter<FnColorVectorContract>
{ 
  static bool visit(FnColorVectorContract op, const LockTag & t) 
    { 
      // t.ossCode << "FnColorVectorContract()";
      return true;
    }
};

// FnColorCrossProduct
template <>
struct OpVisitor<FnColorCrossProduct, LockTag> : public BracketPrinter<FnColorCrossProduct>
{ 
  static bool visit(FnColorCrossProduct op, const LockTag & t) 
    { 
      // t.ossCode << "FnColorCrossProduct()";
      return true;
    }
};

// FnLocalInnerProduct
template <>
struct OpVisitor<FnLocalInnerProduct, LockTag> : public BracketPrinter<FnLocalInnerProduct>
{ 
  static bool visit(FnLocalInnerProduct op, const LockTag & t) 
    { 
      // t.ossCode << "FnLocalInnerProduct()";
      return true;
    }
};

// FnLocalInnerProductReal
template <>
struct OpVisitor<FnLocalInnerProductReal, LockTag> : public BracketPrinter<FnLocalInnerProductReal>
{ 
  static bool visit(FnLocalInnerProductReal op, const LockTag & t) 
    { 
      // t.ossCode << "FnLocalInnerProductReal()";
      return true;
    }
};

// FnQuarkContract13
template <>
struct OpVisitor<FnQuarkContract13, LockTag> : public BracketPrinter<FnQuarkContract13>
{ 
  static bool visit(FnQuarkContract13 op, const LockTag & t) 
    { 
      // t.ossCode << "FnQuarkContract13()";
      return true;
    }
};

// FnQuarkContract14
template <>
struct OpVisitor<FnQuarkContract14, LockTag> : public BracketPrinter<FnQuarkContract14>
{ 
  static bool visit(FnQuarkContract14 op, const LockTag & t) 
    { 
      // t.ossCode << "FnQuarkContract14()";
      return true;
    }
};

// FnQuarkContract23
template <>
struct OpVisitor<FnQuarkContract23, LockTag> : public BracketPrinter<FnQuarkContract23>
{ 
  static bool visit(FnQuarkContract23 op, const LockTag & t) 
    { 
      // t.ossCode << "FnQuarkContract23()";
      return true;
    }
};

// FnQuarkContract24
template <>
struct OpVisitor<FnQuarkContract24, LockTag> : public BracketPrinter<FnQuarkContract24>
{ 
  static bool visit(FnQuarkContract24 op, const LockTag & t) 
    { 
      // t.ossCode << "FnQuarkContract24()";
      return true;
    }
};

// FnQuarkContract12
template <>
struct OpVisitor<FnQuarkContract12, LockTag> : public BracketPrinter<FnQuarkContract12>
{ 
  static bool visit(FnQuarkContract12 op, const LockTag & t) 
    { 
      // t.ossCode << "FnQuarkContract12()";
      return true;
    }
};

// FnQuarkContract34
template <>
struct OpVisitor<FnQuarkContract34, LockTag> : public BracketPrinter<FnQuarkContract34>
{ 
  static bool visit(FnQuarkContract34 op, const LockTag & t) 
    { 
      // t.ossCode << "FnQuarkContract34()";
      return true;
    }
};

// FnColorContract
template <>
struct OpVisitor<FnColorContract, LockTag> : public BracketPrinter<FnColorContract>
{ 
  static bool visit(FnColorContract op, const LockTag & t) 
    { 
      // t.ossCode << "FnColorContract()";
      return true;
    }
};

