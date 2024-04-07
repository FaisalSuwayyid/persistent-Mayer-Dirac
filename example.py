import persistent_mayer_dirac

points =[[1, 1], [7, 0], [4, 6], [9, 6], [0, 14], [2, 19], [9, 17]];

N = 5
dm = 4
maxrad = 7
numfilts = 28

ylabels = []
for i in range(N):
    for q in range(1, N):
            ylabels.append('$\\eta(D_{%d, %d,%d})$' % (len(np_q_th_sequence(N, q, length = dm + 1, sequence = [i], seq = i)) - 1,i,q))
# ylabels

usedlabels = []
usedlocations = []
locs = np.arange(len(radii))
labs = np.round(radii, decimals=3)
for i in range(len(radii)):
    if 2*i < len(radii):
        usedlabels.append(labs[2*i])
        usedlocations.append(locs[2*i])

radii, filts_bettis,\
            filts_means, filts_sums, filts_mins,\
            filts_maxs, filts_stds, filts_gen_means,\
            filts_sec_moms, filts_zets2, filts_zets1,\
            filts_c1, filts_EPs, filts_stns, filts_QWs,\
            d_filts_bettis, d_filts_means, d_filts_sums,\
            d_filts_mins, d_filts_maxs, d_filts_stds,\
            d_filts_gen_means, d_filts_sec_moms, d_filts_zets2,\
            d_filts_zets1, d_filts_QWs = np_filtration(points,\
                                                             num_filtrations = numfilts,\
                                                             max_radius = maxrad,\
                                                             max_dimension = dm, N = N)
vs1_B = flattener_indivisual([filts_bettis,\
            filts_means, filts_sums, filts_mins,\
            filts_maxs, filts_stds, filts_gen_means,\
            filts_sec_moms, filts_zets2, filts_zets1,\
            filts_c1, filts_EPs, filts_stns, filts_QWs,\
            d_filts_bettis, d_filts_means, d_filts_sums,\
            d_filts_mins, d_filts_maxs, d_filts_stds,\
            d_filts_gen_means, d_filts_sec_moms, d_filts_zets2,\
            d_filts_zets1, d_filts_QWs], fil_length = len(radii))


from mpl_toolkits.axes_grid1 import make_axes_locatable

fontsize = 12
vs = vs1_B[14]
fig, ax = plt.subplots()
# fig.figsize(3,4)
# np.arange(100).reshape((10, 10))
im = ax.imshow(vs.T)
plt.yticks(ticks = np.arange(len(ylabels)), labels=ylabels, fontsize = fontsize)
plt.xticks(ticks = np.array(usedlocations).flatten(), labels=np.array(usedlabels).flatten(), rotation = 55, fontsize = fontsize)

# create an Axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

cbar = plt.colorbar(im, cax=cax)
ticklabs = cbar.ax.get_yticklabels()
cbar.ax.set_yticklabels(ticklabs, fontsize = fontsize)

plt.savefig("vs2_B_nullity_N_%s.svg" % (N))
plt.show()
