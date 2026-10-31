# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Bundled curve mnemonic / alias table for header OCR.

Canonical mnemonics follow the CWLS LAS 2.0 ``~Curve`` section conventions and the common
SPWLA / vendor mnemonic usage (GR, SP, CALI, RHOB, NPHI, DT, ILD, ...). Aliases include vendor
variants and the long printed track-header phrases found on analog logs ("COMPENSATED NEUTRON
POROSITY", "GAMMA RAY", "INTERVAL TRANSIT TIME"). ``units`` lists printed unit spellings mapped
to a canonical LAS unit string; ``scale_type`` is the usual presentation (resistivity: log).
"""
from __future__ import annotations

# canonical -> dict(aliases, phrases, unit, scale_type, family)
CURVES: dict[str, dict] = {
    "GR":   dict(family="gamma", unit="GAPI", scale_type="linear",
                 aliases=["GR", "GRC", "SGR", "CGR", "GAM", "GAMMA", "GRGC"],
                 phrases=["GAMMA RAY", "GAMMA RAY API UNITS", "GAMMA RAY API", "NATURAL GAMMA RAY",
                          "GAMMA RAY UNITS", "RADIOACTIVITY GAMMA RAY"]),
    "SP":   dict(family="sp", unit="MV", scale_type="linear",
                 aliases=["SP", "SSP"],
                 phrases=["SPONTANEOUS POTENTIAL", "SELF POTENTIAL", "SPONTANEOUS POTENTIAL MILLIVOLTS"]),
    "CALI": dict(family="caliper", unit="IN", scale_type="linear",
                 aliases=["CALI", "CAL", "CALS", "HCAL", "CALIPER"],
                 phrases=["CALIPER", "HOLE DIAMETER", "CALIPER HOLE DIAMETER", "HOLE DIAM",
                          "CALIPER DIAM IN INCHES", "BOREHOLE DIAMETER"]),
    "RHOB": dict(family="density", unit="G/CC", scale_type="linear",
                 aliases=["RHOB", "RHOZ", "DEN", "ZDEN", "DENS"],
                 phrases=["BULK DENSITY", "FORMATION DENSITY", "COMPENSATED FORMATION DENSITY",
                          "COMPENSATED DENSITY", "BULK DENSITY G CC"]),
    "DRHO": dict(family="density", unit="G/CC", scale_type="linear",
                 aliases=["DRHO", "ZCOR"],
                 phrases=["DENSITY CORRECTION", "BULK DENSITY CORRECTION"]),
    "DPHI": dict(family="porosity", unit="V/V", scale_type="linear",
                 aliases=["DPHI", "DPOR", "PHID", "DPHZ"],
                 phrases=["DENSITY POROSITY", "FORMATION DENSITY POROSITY",
                          "COMPENSATED FORMATION DENSITY POROSITY"]),
    "NPHI": dict(family="porosity", unit="V/V", scale_type="linear",
                 aliases=["NPHI", "CNL", "NPOR", "TNPH", "PHIN", "CNPOR", "SNP", "NEUT"],
                 phrases=["COMPENSATED NEUTRON POROSITY", "NEUTRON POROSITY", "COMPENSATED NEUTRON",
                          "SIDEWALL NEUTRON POROSITY", "NEUTRON POROSITY INDEX", "NEUTRON"]),
    "PE":   dict(family="pe", unit="B/E", scale_type="linear",
                 aliases=["PE", "PEF", "PEFZ"],
                 phrases=["PHOTOELECTRIC FACTOR", "PHOTOELECTRIC EFFECT", "PHOTO ELECTRIC"]),
    "DT":   dict(family="sonic", unit="US/F", scale_type="linear",
                 aliases=["DT", "AC", "DTC", "SONIC", "DELT", "BHC"],
                 phrases=["INTERVAL TRANSIT TIME", "SONIC TRANSIT TIME", "TRANSIT TIME",
                          "DELTA T", "ACOUSTIC", "BHC SONIC", "MICROSECONDS PER FOOT",
                          "ACOUSTILOG", "ACOUSTIC VELOCITY"]),
    "ILD":  dict(family="resistivity", unit="OHMM", scale_type="log",
                 aliases=["ILD", "RILD", "IDPH", "RID"],
                 phrases=["DEEP INDUCTION", "INDUCTION DEEP", "DEEP INDUCTION RESISTIVITY",
                          "INDUCTION LOG DEEP", "ILD DEEP"]),
    "ILM":  dict(family="resistivity", unit="OHMM", scale_type="log",
                 aliases=["ILM", "RILM", "IMPH", "RIM"],
                 phrases=["MEDIUM INDUCTION", "INDUCTION MEDIUM", "ILM MEDIUM"]),
    "LLD":  dict(family="resistivity", unit="OHMM", scale_type="log",
                 aliases=["LLD", "RLLD"], phrases=["DEEP LATEROLOG", "LATEROLOG DEEP"]),
    "LLS":  dict(family="resistivity", unit="OHMM", scale_type="log",
                 aliases=["LLS", "RLLS"], phrases=["SHALLOW LATEROLOG", "LATEROLOG SHALLOW"]),
    "SFL":  dict(family="resistivity", unit="OHMM", scale_type="log",
                 aliases=["SFL", "SFLU", "SFLA", "LL8", "LL3"],
                 phrases=["SPHERICALLY FOCUSED", "SPHERICALLY FOCUSED LOG", "FOCUSED LOG", "GUARD"]),
    "MSFL": dict(family="resistivity", unit="OHMM", scale_type="log",
                 aliases=["MSFL", "MLL", "RXO", "MICROLATEROLOG"],
                 phrases=["MICRO SPHERICALLY FOCUSED", "MICROLATEROLOG", "FLUSHED ZONE RESISTIVITY"]),
    "RES":  dict(family="resistivity", unit="OHMM", scale_type="log",
                 aliases=["RES", "RT", "RESISTIVITY", "INDUCTION"],
                 phrases=["RESISTIVITY", "RESISTIVITY OHMS M2 M", "OHMS M2 M", "INDUCTION",
                          "LONG NORMAL", "SHORT NORMAL", "AMPLIFIED SHORT NORMAL", "LATERAL",
                          "INDUCTION RESISTIVITY", "OHM METERS"]),
    "COND": dict(family="conductivity", unit="MMHO/M", scale_type="linear",
                 aliases=["COND", "CILD", "CIL"],
                 phrases=["CONDUCTIVITY", "MILLIMHOS PER METER", "MILLIMHOS M", "INDUCTION CONDUCTIVITY"]),
}

# printed unit spelling (normalised, uppercase, no punctuation except /) -> canonical LAS unit
UNITS: dict[str, str] = {
    "API": "GAPI", "API UNITS": "GAPI", "APL UNITS": "GAPI", "API UNIT": "GAPI", "GAPI": "GAPI", "API GR UNITS": "GAPI",
    "MV": "MV", "MILLIVOLTS": "MV", "MILLIVOLT": "MV",
    "IN": "IN", "INCHES": "IN", "INCH": "IN", "IN INCHES": "IN",
    "G/CC": "G/CC", "GM/CC": "G/CC", "G/CM3": "G/CC", "GMS/CC": "G/CC", "GRAMS/CC": "G/CC", "G/C3": "G/CC",
    "%": "%", "PERCENT": "%", "PU": "%", "P.U.": "%", "POROSITY INDEX %": "%",
    "US/FT": "US/F", "US/F": "US/F", "USEC/FT": "US/F", "MICROSEC/FT": "US/F", "MICROSECONDS/FT": "US/F",
    "MICROSECONDS PER FOOT": "US/F", "μSEC/FT": "US/F",
    "OHM-M": "OHMM", "OHMM": "OHMM", "OHMS M2/M": "OHMM", "OHMS-M2/M": "OHMM", "OHM-METERS": "OHMM",
    "OHMS": "OHMM", "OHM METERS": "OHMM", "OHMS M": "OHMM",
    "MMHOS/M": "MMHO/M", "MMHO/M": "MMHO/M", "MILLIMHOS/M": "MMHO/M", "MILLIMHOS": "MMHO/M",
    "B/E": "B/E", "BARNS/ELECTRON": "B/E",
}

# LAS unit spellings in files -> canonical (for evaluating against paired LAS headers)
LAS_UNIT_ALIASES: dict[str, str] = {
    "GAPI": "GAPI", "API": "GAPI", "MV": "MV", "IN": "IN", "INCH": "IN",
    "G/C3": "G/CC", "G/CC": "G/CC", "GM/CC": "G/CC", "K/M3": "G/CC",
    "V/V": "V/V", "DEC": "V/V", "%": "%", "PU": "%", "PERCENT": "%",
    "US/F": "US/F", "US/FT": "US/F", "USEC/FT": "US/F", "OHMM": "OHMM", "OHM-M": "OHMM", "OHMS": "OHMM",
    "MMHO/M": "MMHO/M", "B/E": "B/E",
}

# porosity curves are printed either as fraction (V/V) or percent; treat both as compatible
COMPATIBLE_UNITS = [{"V/V", "%"}]
