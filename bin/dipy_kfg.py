# Basic imports
import time
import numpy as np
import nibabel as nib

# Import tracking stuff
from dipy.tracking.localtrack import ThresholdTissueClassifier, local_tracker
from dipy.tracking.local import LocalTracking
from dipy.reconst.dg import ProbabilisticOdfWightedDirectionGetter
from dipy.tracking import utils

#Import gui stuff
import traits.api as T
from traitsui.api import *

# Import a few different models as examples
from dipy.reconst.shm import CsaOdfModel
from dipy.core.sphere import HemiSphere
from dipy.reconst.peaks import peaks_from_model, default_sphere
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response

# Import data stuff
from dipy.segment.mask import median_otsu
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere

#set up options
#all_tracking_type = {'Probabilistic':prob_tracking_example,'Deterministic':detr_tracking_example}
#all_shmodels = {'CsaOdfModel':CsaOdfModel, 'CsdOdfModel':ConstrainedSphericalDeconvModel}

def clipMask(mask):
    out = mask.copy()
    index = [slice(None)] * out.ndim
    for i in range(len(index)):
        idx = index[:]
        idx[i] = [0, -1]
        out[idx] = 0.
    return out


def data_pleez():
    fetch_stanford_hardi()
    img, gtab = read_stanford_hardi()
    data = img.get_data()
    affine = img.get_affine()
    sphere = default_sphere
    return img, gtab, data, affine, sphere

def make_mask(data):
    _, mask = median_otsu(data, 3, 1, False, vol_idx=range(10, 50))
    mask = clipMask(mask)
    return mask

def make_hdr(mask,img):
    hdr = nib.trackvis.empty_header()
    hdr['dim'] = mask.shape
    hdr['voxel_size'] = img.get_header().get_zooms()[:3]
    hdr['voxel_order'] = 'ras'
    return hdr

def make_tkvis_aff(hdr):
    trackvis_affine = utils.affine_for_trackvis(hdr['voxel_size'])
    return trackvis_affine

#def make_seeds(tar_numseeds,gfa):
#    seeds = utils.seeds_from_mask(gfa > .25, 2, affine=affine)
#    seeds = seeds[::len(seeds) // N + 1]
#    return seeds

def make_model(model_type,gtab,data):
    if model_type=='CsaOdfModel':
        model = CsaOdfModel(gtab, 8)
        save_name = "test_csa.trk"
    else:
        small_sphere = HemiSphere.from_sphere(get_sphere("symmetric362"))
        r, _ = auto_response(gtab, data)
        model = ConstrainedSphericalDeconvModel(gtab, r, sh_order=10,reg_sphere=small_sphere,lambda_=np.sqrt(1./2))
        save_name = "test_csd.trk"
    return model, save_name

class NewInterface(T.HasTraits): #inherits from HasTraits class
    trktype = T.Enum('probabilistic','deterministic',desc="what type of tracking will be used", label="Choose tracking method", )
    model_type = T.Enum('CsaOdfModel','CsdOdfModel',desc="which model will be used to reconstruct the fiber population in a given voxel",label="Choose model", )
    target_numseeds = T.CInt(5000,desc="the total number of seeds that will initialize the tracking algorithm",label="Target # Seeds", )

    #remember need setup to run on button press

    def _setup(self):
        img, gtab, data, affine, sphere = data_pleez()
        mask = make_mask(data)
        hdr = make_hdr(mask,img)
        trackvis_affine = make_tkvis_aff(hdr)
        #seeds = make_seeds(self.target_numseeds) #need gfa defined.. gfa = general FA
        Nseeds = self.target_numseeds
        model, save_name = make_model(self.model_type,gtab,data)
        track_type = self.trktype
        return model, track_type, data, mask, Nseeds, hdr, save_name,sphere,affine,trackvis_affine

    def track(self):
        model, track_type, data, mask, Nseeds, hdr, save_name,sphere,affine,trackvis_affine = newint._setup()
        print "we have chosen the %s model, and will save the %s tracks as %s" % (model, track_type, save_name)
        csapeaks = peaks_from_model(model=model,data=data,sphere=sphere,relative_peak_threshold=.5,min_separation_angle=45,mask=mask,return_odf=False,normalize_peaks=True) #how does it see sphere/aff
        gfa = csapeaks.gfa
        gfa = np.where(np.isnan(gfa), 0., gfa)
        ttc = ThresholdTissueClassifier(gfa, .2)

        # Create around N seeds
        seeds = utils.seeds_from_mask(gfa > .25, 2, affine=affine)
        seeds = seeds[::len(seeds) // Nseeds + 1]

        # Create streamline generator
        streamlines = LocalTracking(csapeaks, ttc, seeds, affine, .5, max_cross=1)
        trk_streamlines = utils.move_streamlines(streamlines,input_space=affine,output_space=trackvis_affine)
        trk = ((streamline, None, None) for streamline in trk_streamlines)

        # Save streamlines
        nib.trackvis.write(save_name, trk, hdr)

def main():
    newint = NewInterface()
    newint.configure_traits()
    newint.track()

if __name__ == "__main__":
    main()

