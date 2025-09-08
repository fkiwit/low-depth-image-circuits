import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

plt.rc ('font', size = 15) # steuert die Standardtextgröße
plt.rc ('axes', labelsize = 15) # Schriftgröße der x- und y-Beschriftungen
plt.rc ('xtick', labelsize = 12) #Schriftgröße der x-Tick-Labels
plt.rc ('ytick', labelsize = 12) #Schriftgröße der y-Tick-Labels
plt.rc ('legend', fontsize = 12) #Schriftgröße der Legende

font_dirs = ['fonts']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

plt.rcParams['font.family'] = 'Times New Roman'   # normal text in TNR
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

color_palette = {
    'blue': (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
    'red': (0.8352941176470589, 0.3686274509803922, 0.0),
    'green': (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
    'orange': (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
    'pink': (0.8,  0.47058823529411764,  0.7372549019607844), 
    'yellow': (0.9254901960784314, 0.8823529411764706, 0.2),
    'gray': (0.9, 0.9, 0.9)
    }

marker = {
    "circle": "o",
    "diamond": "d",
    "square": "s",
    "triangle": "^",
    "triangle_down": "v",
}

data_labels = {
    "fashion_mnist": "Fashion MNIST",
    "cifar10": "CIFAR-10"
}

marker_size = 8
line_width = 2
markeredgewidth = 1
handlelength = 1.5
