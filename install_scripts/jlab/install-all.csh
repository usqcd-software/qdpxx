#!/bin/tcsh

ssh -n qcd3g-i01 "cd qcd/qdp++; ./install.csh"
ssh -n qcd3g-i02 "cd qcd/qdp++; ./install.csh"
ssh -n qcdi01    "cd qcd/qdp++; ./install.csh"
ssh -n qcdi02    "cd qcd/qdp++; ./install.csh"
