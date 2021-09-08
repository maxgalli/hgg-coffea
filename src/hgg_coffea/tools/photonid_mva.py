import warnings
from typing import List, Optional, Tuple

import awkward
import numpy
import xgboost


def load_photonid_mva(fname: str) -> Optional[xgboost.Booster]:
    try:
        photonid_mva = xgboost.Booster()
        photonid_mva.load_model(fname)
    except xgboost.core.XGBoostError:
        warnings.warn(f"SKIPPING photonid_mva, could not find: {fname}")
        photonid_mva = None
    return photonid_mva


def calculate_photonid_mva(
    mva: Tuple[Optional[xgboost.Booster], List[str]],
    photon: awkward.Array,
) -> awkward.Array:
    """
    Ordered list of features
    EB: 
        events.Photon.energyRaw
        events.Photon.r9
        events.Photon.sieie
        events.Photon.etaWidth
        events.Photon.phiWidth
        events.Photon.sieip
        events.Photon.s4
        events.Photon.pfPhoIso03
        events.Photon.pfChargedIsoPFPV
        events.Photon.pfChargedIsoWorstVtx
        events.Photon.eta
        events.fixedGridRhoAll
    
    EE: add

    """
    photonid_mva, var_order = mva

    bdt_inputs = {}
    bdt_inputs = numpy.column_stack(
        [awkward.to_numpy(awkward.flatten(photon[name])) for name in var_order]
        )
    tempmatrix = xgboost.DMatrix(bdt_inputs, feature_names=var_order)

    counts = awkward.num(photon, axis=-1)
    photon["mvaID_recomputed"] = awkward.unflatten(photonid_mva.predict(tempmatrix), counts)

    return photon
