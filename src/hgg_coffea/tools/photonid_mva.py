from typing import List, Optional, Tuple

import awkward
import numpy
import xgboost


def calculate_photonid_mva(
    mva: Tuple[Optional[xgboost.Booster], List[str]],
    photon: awkward.Array,
) -> awkward.Array:
    """ Recompute PhotonIDMVA on-the-fly. This step is necessary considering that the inputs have to be corrected
    with the QRC process. Following is the list of features (barrel has 12, endcap two more): 
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
    
    EE: todo
    """
    photonid_mva, var_order = mva

    if photonid_mva is None:
        return awkward.ones_like(photon)

    bdt_inputs = {}
    bdt_inputs = numpy.column_stack(
        [awkward.to_numpy(awkward.flatten(photon[name])) for name in var_order]
        )
    tempmatrix = xgboost.DMatrix(bdt_inputs, feature_names=var_order)

    counts = awkward.num(photon, axis=-1)
    photon["mvaID"] = awkward.unflatten(photonid_mva.predict(tempmatrix), counts)

    photon["mvaID"] = -numpy.log(1./photon["mvaID"] - 1.)
    photon["mvaID"] = 2. / (1. + numpy.exp(-2.*photon["mvaID"])) - 1.    

    return mvaID
