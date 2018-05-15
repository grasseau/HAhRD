import matplotlib.pyplot as plt
from matplotlib import cm
from descartes.patch import PolygonPatch
from geometry.panels import generate_modules, modules_to_panels

module_to_panel, panel_to_modules = modules_to_panels(wafer_size=19.041, grid_size=13)
modules = generate_modules(wafer_size=19.041, grid_size=13)
modules_dict = {mod.id:mod for mod in modules}
colors = [cm.get_cmap('Vega20')(i) for i in range(20)]
colors.extend([cm.get_cmap('Set3')(i) for i in range(12)])
colors.extend([cm.get_cmap('Dark2')(i) for i in range(8)])
fig = plt.figure()
ax = fig.add_subplot(111)
# Plot with different colors groups of modules mapped to different panels
for i,(panel,panel_modules) in enumerate(panel_to_modules.items()):
    color = colors[i%len(colors)]
    for module in panel_modules:
        patch = PolygonPatch(modules_dict[module].vertices, facecolor=color, edgecolor='#AAAAAA',  zorder=1)
        ax.add_patch(patch)
# Plot module borders
for module in modules:
    x, y = module.vertices.exterior.xy
    ax.plot(x, y, color='#AAAAAA', zorder=1)
plt.rcParams['savefig.dpi'] = 1000
fig.savefig('test_panel_mapping.svg')
