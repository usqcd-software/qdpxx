// FnArcCos
template <>
struct OpVisitor<FnArcCos, ParseTag> : public BracketPrinter<FnArcCos>
{ 
  static bool visit(FnArcCos op, const ParseTag & t) 
    { 
      t.ossCode << "FnArcCos()";
      return true;
    }
};

// FnArcSin
template <>
struct OpVisitor<FnArcSin, ParseTag> : public BracketPrinter<FnArcSin>
{ 
  static bool visit(FnArcSin op, const ParseTag & t) 
    { 
      t.ossCode << "FnArcSin()";
      return true;
    }
};

// FnArcTan
template <>
struct OpVisitor<FnArcTan, ParseTag> : public BracketPrinter<FnArcTan>
{ 
  static bool visit(FnArcTan op, const ParseTag & t) 
    { 
      t.ossCode << "FnArcTan()";
      return true;
    }
};

// FnCeil
template <>
struct OpVisitor<FnCeil, ParseTag> : public BracketPrinter<FnCeil>
{ 
  static bool visit(FnCeil op, const ParseTag & t) 
    { 
      t.ossCode << "FnCeil()";
      return true;
    }
};

// FnCos
template <>
struct OpVisitor<FnCos, ParseTag> : public BracketPrinter<FnCos>
{ 
  static bool visit(FnCos op, const ParseTag & t) 
    { 
      t.ossCode << "FnCos()";
      return true;
    }
};

// FnHypCos
template <>
struct OpVisitor<FnHypCos, ParseTag> : public BracketPrinter<FnHypCos>
{ 
  static bool visit(FnHypCos op, const ParseTag & t) 
    { 
      t.ossCode << "FnHypCos()";
      return true;
    }
};

// FnExp
template <>
struct OpVisitor<FnExp, ParseTag> : public BracketPrinter<FnExp>
{ 
  static bool visit(FnExp op, const ParseTag & t) 
    { 
      t.ossCode << "FnExp()";
      return true;
    }
};

// FnFabs
template <>
struct OpVisitor<FnFabs, ParseTag> : public BracketPrinter<FnFabs>
{ 
  static bool visit(FnFabs op, const ParseTag & t) 
    { 
      t.ossCode << "FnFabs()";
      return true;
    }
};

// FnFloor
template <>
struct OpVisitor<FnFloor, ParseTag> : public BracketPrinter<FnFloor>
{ 
  static bool visit(FnFloor op, const ParseTag & t) 
    { 
      t.ossCode << "FnFloor()";
      return true;
    }
};

// FnLog
template <>
struct OpVisitor<FnLog, ParseTag> : public BracketPrinter<FnLog>
{ 
  static bool visit(FnLog op, const ParseTag & t) 
    { 
      t.ossCode << "FnLog()";
      return true;
    }
};

// FnLog10
template <>
struct OpVisitor<FnLog10, ParseTag> : public BracketPrinter<FnLog10>
{ 
  static bool visit(FnLog10 op, const ParseTag & t) 
    { 
      t.ossCode << "FnLog10()";
      return true;
    }
};

// FnSin
template <>
struct OpVisitor<FnSin, ParseTag> : public BracketPrinter<FnSin>
{ 
  static bool visit(FnSin op, const ParseTag & t) 
    { 
      t.ossCode << "FnSin()";
      return true;
    }
};

// FnHypSin
template <>
struct OpVisitor<FnHypSin, ParseTag> : public BracketPrinter<FnHypSin>
{ 
  static bool visit(FnHypSin op, const ParseTag & t) 
    { 
      t.ossCode << "FnHypSin()";
      return true;
    }
};

// FnSqrt
template <>
struct OpVisitor<FnSqrt, ParseTag> : public BracketPrinter<FnSqrt>
{ 
  static bool visit(FnSqrt op, const ParseTag & t) 
    { 
      t.ossCode << "FnSqrt()";
      return true;
    }
};

// FnTan
template <>
struct OpVisitor<FnTan, ParseTag> : public BracketPrinter<FnTan>
{ 
  static bool visit(FnTan op, const ParseTag & t) 
    { 
      t.ossCode << "FnTan()";
      return true;
    }
};

// FnHypTan
template <>
struct OpVisitor<FnHypTan, ParseTag> : public BracketPrinter<FnHypTan>
{ 
  static bool visit(FnHypTan op, const ParseTag & t) 
    { 
      t.ossCode << "FnHypTan()";
      return true;
    }
};

// OpUnaryMinus
template <>
struct OpVisitor<OpUnaryMinus, ParseTag> : public BracketPrinter<OpUnaryMinus>
{ 
  static bool visit(OpUnaryMinus op, const ParseTag & t) 
    { 
      t.ossCode << "OpUnaryMinus()";
      return true;
    }
};

// OpUnaryPlus
template <>
struct OpVisitor<OpUnaryPlus, ParseTag> : public BracketPrinter<OpUnaryPlus>
{ 
  static bool visit(OpUnaryPlus op, const ParseTag & t) 
    { 
      t.ossCode << "OpUnaryPlus()";
      return true;
    }
};

// OpBitwiseNot
template <>
struct OpVisitor<OpBitwiseNot, ParseTag> : public BracketPrinter<OpBitwiseNot>
{ 
  static bool visit(OpBitwiseNot op, const ParseTag & t) 
    { 
      t.ossCode << "OpBitwiseNot()";
      return true;
    }
};

// OpIdentity
template <>
struct OpVisitor<OpIdentity, ParseTag> : public BracketPrinter<OpIdentity>
{ 
  static bool visit(OpIdentity op, const ParseTag & t) 
    { 
      t.ossCode << "OpIdentity()";
      return true;
    }
};

// OpNot
template <>
struct OpVisitor<OpNot, ParseTag> : public BracketPrinter<OpNot>
{ 
  static bool visit(OpNot op, const ParseTag & t) 
    { 
      t.ossCode << "OpNot()";
      return true;
    }
};

// OpAdd
template <>
struct OpVisitor<OpAdd, ParseTag> : public BracketPrinter<OpAdd>
{ 
  static bool visit(OpAdd op, const ParseTag & t) 
    { 
      t.ossCode << "OpAdd()";
      return true;
    }
};

// OpSubtract
template <>
struct OpVisitor<OpSubtract, ParseTag> : public BracketPrinter<OpSubtract>
{ 
  static bool visit(OpSubtract op, const ParseTag & t) 
    { 
      t.ossCode << "OpSubtract()";
      return true;
    }
};

// OpMultiply
template <>
struct OpVisitor<OpMultiply, ParseTag> : public BracketPrinter<OpMultiply>
{ 
  static bool visit(OpMultiply op, const ParseTag & t) 
    { 
      t.ossCode << "OpMultiply()";
      return true;
    }
};

// OpDivide
template <>
struct OpVisitor<OpDivide, ParseTag> : public BracketPrinter<OpDivide>
{ 
  static bool visit(OpDivide op, const ParseTag & t) 
    { 
      t.ossCode << "OpDivide()";
      return true;
    }
};

// OpMod
template <>
struct OpVisitor<OpMod, ParseTag> : public BracketPrinter<OpMod>
{ 
  static bool visit(OpMod op, const ParseTag & t) 
    { 
      t.ossCode << "OpMod()";
      return true;
    }
};

// OpBitwiseAnd
template <>
struct OpVisitor<OpBitwiseAnd, ParseTag> : public BracketPrinter<OpBitwiseAnd>
{ 
  static bool visit(OpBitwiseAnd op, const ParseTag & t) 
    { 
      t.ossCode << "OpBitwiseAnd()";
      return true;
    }
};

// OpBitwiseOr
template <>
struct OpVisitor<OpBitwiseOr, ParseTag> : public BracketPrinter<OpBitwiseOr>
{ 
  static bool visit(OpBitwiseOr op, const ParseTag & t) 
    { 
      t.ossCode << "OpBitwiseOr()";
      return true;
    }
};

// OpBitwiseXor
template <>
struct OpVisitor<OpBitwiseXor, ParseTag> : public BracketPrinter<OpBitwiseXor>
{ 
  static bool visit(OpBitwiseXor op, const ParseTag & t) 
    { 
      t.ossCode << "OpBitwiseXor()";
      return true;
    }
};

// FnLdexp
template <>
struct OpVisitor<FnLdexp, ParseTag> : public BracketPrinter<FnLdexp>
{ 
  static bool visit(FnLdexp op, const ParseTag & t) 
    { 
      t.ossCode << "FnLdexp()";
      return true;
    }
};

// FnPow
template <>
struct OpVisitor<FnPow, ParseTag> : public BracketPrinter<FnPow>
{ 
  static bool visit(FnPow op, const ParseTag & t) 
    { 
      t.ossCode << "FnPow()";
      return true;
    }
};

// FnFmod
template <>
struct OpVisitor<FnFmod, ParseTag> : public BracketPrinter<FnFmod>
{ 
  static bool visit(FnFmod op, const ParseTag & t) 
    { 
      t.ossCode << "FnFmod()";
      return true;
    }
};

// FnArcTan2
template <>
struct OpVisitor<FnArcTan2, ParseTag> : public BracketPrinter<FnArcTan2>
{ 
  static bool visit(FnArcTan2 op, const ParseTag & t) 
    { 
      t.ossCode << "FnArcTan2()";
      return true;
    }
};

// OpLT
template <>
struct OpVisitor<OpLT, ParseTag> : public BracketPrinter<OpLT>
{ 
  static bool visit(OpLT op, const ParseTag & t) 
    { 
      t.ossCode << "OpLT()";
      return true;
    }
};

// OpLE
template <>
struct OpVisitor<OpLE, ParseTag> : public BracketPrinter<OpLE>
{ 
  static bool visit(OpLE op, const ParseTag & t) 
    { 
      t.ossCode << "OpLE()";
      return true;
    }
};

// OpGT
template <>
struct OpVisitor<OpGT, ParseTag> : public BracketPrinter<OpGT>
{ 
  static bool visit(OpGT op, const ParseTag & t) 
    { 
      t.ossCode << "OpGT()";
      return true;
    }
};

// OpGE
template <>
struct OpVisitor<OpGE, ParseTag> : public BracketPrinter<OpGE>
{ 
  static bool visit(OpGE op, const ParseTag & t) 
    { 
      t.ossCode << "OpGE()";
      return true;
    }
};

// OpEQ
template <>
struct OpVisitor<OpEQ, ParseTag> : public BracketPrinter<OpEQ>
{ 
  static bool visit(OpEQ op, const ParseTag & t) 
    { 
      t.ossCode << "OpEQ()";
      return true;
    }
};

// OpNE
template <>
struct OpVisitor<OpNE, ParseTag> : public BracketPrinter<OpNE>
{ 
  static bool visit(OpNE op, const ParseTag & t) 
    { 
      t.ossCode << "OpNE()";
      return true;
    }
};

// OpAnd
template <>
struct OpVisitor<OpAnd, ParseTag> : public BracketPrinter<OpAnd>
{ 
  static bool visit(OpAnd op, const ParseTag & t) 
    { 
      t.ossCode << "OpAnd()";
      return true;
    }
};

// OpOr
template <>
struct OpVisitor<OpOr, ParseTag> : public BracketPrinter<OpOr>
{ 
  static bool visit(OpOr op, const ParseTag & t) 
    { 
      t.ossCode << "OpOr()";
      return true;
    }
};

// OpLeftShift
template <>
struct OpVisitor<OpLeftShift, ParseTag> : public BracketPrinter<OpLeftShift>
{ 
  static bool visit(OpLeftShift op, const ParseTag & t) 
    { 
      t.ossCode << "OpLeftShift()";
      return true;
    }
};

// OpRightShift
template <>
struct OpVisitor<OpRightShift, ParseTag> : public BracketPrinter<OpRightShift>
{ 
  static bool visit(OpRightShift op, const ParseTag & t) 
    { 
      t.ossCode << "OpRightShift()";
      return true;
    }
};

// OpAssign
template <>
struct OpVisitor<OpAssign, ParseTag> : public BracketPrinter<OpAssign>
{ 
  static bool visit(OpAssign op, const ParseTag & t) 
    { 
      t.ossCode << "OpAssign()";
      return true;
    }
};

// OpAddAssign
template <>
struct OpVisitor<OpAddAssign, ParseTag> : public BracketPrinter<OpAddAssign>
{ 
  static bool visit(OpAddAssign op, const ParseTag & t) 
    { 
      t.ossCode << "OpAddAssign()";
      return true;
    }
};

// OpSubtractAssign
template <>
struct OpVisitor<OpSubtractAssign, ParseTag> : public BracketPrinter<OpSubtractAssign>
{ 
  static bool visit(OpSubtractAssign op, const ParseTag & t) 
    { 
      t.ossCode << "OpSubtractAssign()";
      return true;
    }
};

// OpMultiplyAssign
template <>
struct OpVisitor<OpMultiplyAssign, ParseTag> : public BracketPrinter<OpMultiplyAssign>
{ 
  static bool visit(OpMultiplyAssign op, const ParseTag & t) 
    { 
      t.ossCode << "OpMultiplyAssign()";
      return true;
    }
};

// OpDivideAssign
template <>
struct OpVisitor<OpDivideAssign, ParseTag> : public BracketPrinter<OpDivideAssign>
{ 
  static bool visit(OpDivideAssign op, const ParseTag & t) 
    { 
      t.ossCode << "OpDivideAssign()";
      return true;
    }
};

// OpModAssign
template <>
struct OpVisitor<OpModAssign, ParseTag> : public BracketPrinter<OpModAssign>
{ 
  static bool visit(OpModAssign op, const ParseTag & t) 
    { 
      t.ossCode << "OpModAssign()";
      return true;
    }
};

// OpBitwiseOrAssign
template <>
struct OpVisitor<OpBitwiseOrAssign, ParseTag> : public BracketPrinter<OpBitwiseOrAssign>
{ 
  static bool visit(OpBitwiseOrAssign op, const ParseTag & t) 
    { 
      t.ossCode << "OpBitwiseOrAssign()";
      return true;
    }
};

// OpBitwiseAndAssign
template <>
struct OpVisitor<OpBitwiseAndAssign, ParseTag> : public BracketPrinter<OpBitwiseAndAssign>
{ 
  static bool visit(OpBitwiseAndAssign op, const ParseTag & t) 
    { 
      t.ossCode << "OpBitwiseAndAssign()";
      return true;
    }
};

// OpBitwiseXorAssign
template <>
struct OpVisitor<OpBitwiseXorAssign, ParseTag> : public BracketPrinter<OpBitwiseXorAssign>
{ 
  static bool visit(OpBitwiseXorAssign op, const ParseTag & t) 
    { 
      t.ossCode << "OpBitwiseXorAssign()";
      return true;
    }
};

// OpLeftShiftAssign
template <>
struct OpVisitor<OpLeftShiftAssign, ParseTag> : public BracketPrinter<OpLeftShiftAssign>
{ 
  static bool visit(OpLeftShiftAssign op, const ParseTag & t) 
    { 
      t.ossCode << "OpLeftShiftAssign()";
      return true;
    }
};

// OpRightShiftAssign
template <>
struct OpVisitor<OpRightShiftAssign, ParseTag> : public BracketPrinter<OpRightShiftAssign>
{ 
  static bool visit(OpRightShiftAssign op, const ParseTag & t) 
    { 
      t.ossCode << "OpRightShiftAssign()";
      return true;
    }
};

// FnWhere
template <>
struct OpVisitor<FnWhere, ParseTag> : public BracketPrinter<FnWhere>
{ 
  static bool visit(FnWhere op, const ParseTag & t) 
    { 
      t.ossCode << "FnWhere()";
      return true;
    }
};

