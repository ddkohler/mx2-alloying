all_plot = True
save = True

if False:
    wt.artists.apply_rcparams(kind="publication")

 
# --- PL map summary -----------------------------------------------------------------------------


if all_plot:
    import pl_summary
    pl_summary.run(save)


# --- junction reflection contrast ---------------------------------------------------------------


if all_plot:
    import reflection_contrast as mod
    mod.run(save)


# --- compare PL and Raman maps -------------------------------------------------------------------


if all_plot:
    import pl_vs_raman_spatial
    pl_vs_raman_spatial.run(save)


# --- representative Raman spec vs y --------------------------------------------------------------


if all_plot:
    import raman_vs_y
    raman_vs_y.run(save)


# --- maps of raman modes -------------------------------------------------------------------------


if all_plot:
    import raman_mode_maps
    raman_mode_maps.run(save)


# --- SHG polarimetry image ------------------------------------------------------------------------

if all_plot:
    import shg_polarimetry
    shg_polarimetry.run(save)
    shg_polarimetry.run2(save)


# --- reflection contrast vs SiO2 thickness --------------------------------------------------------


if all_plot:
    import rc_vs_SiO2_thickness as mod
    mod.run(save)


# --- raman SI figure ------------------------------------------------------------------------------


if all_plot:
    import raman_si as mod
    mod.run(save)


# --- cluster analysis -----------------------------------------------------------------------------


if all_plot:
    import separator as mod
    mod.main(save)

