from PyQt5 import QtWidgets ,QtCore, QtGui
from mainwindow import Ui_MainWindow
from scipy import signal
import sys
import scipy
from scipy.io import wavfile
import os
from fpdf import FPDF
import shutil
import pyqtgraph.exporters
import numpy as np
from scipy import fftpack
from scipy.fftpack import fft
import sounddevice as sd 
from pyqtgraph import PlotWidget ,PlotItem
import pyqtgraph as pg 
from collections import OrderedDict

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.radiobutton_number=0
        self.data_equalized=np.array([])
        self.equalizer_sliders = [self.ui.equalizer_slider1, self.ui.equalizer_slider2, self.ui.equalizer_slider3,self.ui.equalizer_slider4,self.ui.equalizer_slider5,self.ui.equalizer_slider6,self.ui.equalizer_slider7,self.ui.equalizer_slider8,self.ui.equalizer_slider9,self.ui.equalizer_slider10]

        for equalizer_slider in self.equalizer_sliders:
            equalizer_slider.valueChanged.connect(self.Equalizer)

        self.color1 = pg.mkPen(color=(255,255,0))
        self.ui.actionopen_file.triggered.connect(lambda :self.loadFile())
        self.radiobuttons=[self.ui.radiobutton1,self.ui.radiobutton2,self.ui.radiobutton3,self.ui.radiobutton4]
        for radiobutton in self.radiobuttons:
            radiobutton.toggled.connect(self.onclicked)
        self.ui.pause.clicked.connect(lambda : self.pause_fn() )
        self.ui.play.clicked.connect(lambda :self.play_fn())
        self.ui.zoomx.clicked.connect(lambda:self.zoom(0.75,1))
        self.ui.zoomy.clicked.connect(lambda:self.zoom(1,0.75))
        self.ui.actionZoom_In.triggered.connect(lambda : self.zoom(0.75,0.75))
        self.ui.actionZoom_Out.triggered.connect(lambda : self.zoom(1.25,1.25))
        self.ui.actionZoom_X.triggered.connect(lambda:self.zoom(0.75,1))
        self.ui.actionZoom_Y.triggered.connect(lambda:self.zoom(1,0.75))
        self.ui.Export.clicked.connect(lambda:self.printPDF())
        self.ui.zoom_in.clicked.connect(lambda : self.zoom(0.75,0.75))
        self.ui.zoom_out.clicked.connect(lambda : self.zoom(1.25,1.25))
        self.ui.clear_Button.clicked.connect(lambda:self.clear())
        self.ui.ExitButton.clicked.connect(exit) 
        self.ui.actionnew_window.triggered.connect(self.new_window)
        self.ui.scroll_x.valueChanged.connect(self.scrollx)
        self.ui.comboBox.activated.connect(self.changecolor)
        self.ui.maxfrequency_slider.valueChanged.connect(self.spectrogram_change)
        self.ui.minfrequency_slider.valueChanged.connect(self.spectrogram_change)
        self.fmin=0
        
        
    def loadFile(self) :
        file_name = QtGui.QFileDialog.getOpenFileName( self, 'choose the signal', os.getenv('HOME') ,"wav(*.wav)" )
        self.path = file_name[0] 
        if self.path =="" :
            return
        self.sampling_frequency , self.data = wavfile.read(self.path)
        samples_count=self.data.shape[0]
        self.time=np.arange(samples_count)/self.sampling_frequency
        self.amplitude=np.array(self.data)
        self.ui.input_graph.plot(self.time,self.amplitude,pen=self.color1)
        self.ui.input_graph.plotItem.setXRange(min(self.time),max(self.time)/50)
        self.ui.input_graph.plotItem.setYRange(min(self.amplitude),max(self.amplitude))
        self.fmax=self.sampling_frequency/2
        self.generate_spectrogram(self.ui.Spectrogram_input,self.amplitude)
        sd.play(self.data,self.sampling_frequency)
        self.ui.output_graph.plot(self.time,self.amplitude,pen=self.color1)
        self.ui.output_graph.plotItem.setXRange(min(self.time),max(self.time)/50)
        self.ui.output_graph.plotItem.setYRange(min(self.amplitude),max(self.amplitude))
        self.generate_spectrogram(self.ui.Spectrogram_output,self.amplitude)
        

        ##fourier
        self.DataFourier = np.fft.fft(self.data) 
        self.phase=np.angle(self.DataFourier)
        self.freqs=np.fft.fftfreq(len(self.data),1/self.sampling_frequency)
        self.Datafourier_amplitude = np.abs( self.DataFourier )
        self.bandlength=int(len(self.freqs)/20)
        self.frequency_length=int(len(self.freqs)/2)

    
        
    def gain_fn (self,data,p, start , end):
        self.gain=[self.ui.equalizer_slider10.value(),self.ui.equalizer_slider9.value(),self.ui.equalizer_slider8.value(),self.ui.equalizer_slider7.value(),self.ui.equalizer_slider6.value(),self.ui.equalizer_slider5.value(),self.ui.equalizer_slider4.value(),self.ui.equalizer_slider3.value(),self.ui.equalizer_slider2.value(),self.ui.equalizer_slider1.value()]

        for i in range(start,end):

            data[self.frequency_length+ i*self.bandlength : self.frequency_length+ (i+1) * self.bandlength]=data[self.frequency_length+ i*self.bandlength : self.frequency_length+ (i+1) * self.bandlength]*np.float64(self.gain[i])*p
            data[self.frequency_length- (i+1)*self.bandlength : self.frequency_length- i*self.bandlength ]=data[self.frequency_length- (i+1)*self.bandlength : self.frequency_length- i*self.bandlength ]*np.float64(self.gain[i])*p
            
    def Equalizer(self):
        
        self.Data_update=self.Datafourier_amplitude.copy()
        self.gain_fn(self.Data_update,p=1,start= 0 ,end= 10)  
        datafourier_modified=np.multiply(self.Data_update,np.exp(1j*self.phase))
        self.data_equalized=np.real(np.fft.ifft(datafourier_modified))
        self.ui.output_graph.clear()
        self.ui.output_graph.plot(self.time,self.data_equalized ,pen=self.color1)
        self.ui.output_graph.plotItem.setXRange(min(self.time),max(self.time)/50)
        self.ui.output_graph.plotItem.setYRange(min(self.data_equalized),max(self.data_equalized))
        self.ui.Spectrogram_output.clear()
        self.generate_spectrogram(self.ui.Spectrogram_output,self.data_equalized)
            
    def spectrogram_change (self):
        
        self.Data_ch=self.Data_update.copy()
        if self.ui.maxfrequency_slider.value() ==0:
            self.fmax=self.sampling_frequency/2 
        elif self.ui.maxfrequency_slider.value() ==1:
            self.fmax=(self.sampling_frequency/2 )-(self.sampling_frequency/20)
            self.gain_fn(self.Data_ch,p=0,start=0,end=1)  
             
        elif self.ui.maxfrequency_slider.value() ==2:
            self.fmax=(self.sampling_frequency/2 )-2*(self.sampling_frequency/20)
            
            self.gain_fn(self.Data_ch,p=0,start=0,end=2)
        elif self.ui.maxfrequency_slider.value() ==3:
            self.fmax=(self.sampling_frequency/2 )-3*(self.sampling_frequency/20)
            self.gain_fn(self.Data_ch,p=0,start=0,end=3)

        if self.ui.minfrequency_slider.value() ==0:
            self.fmin=0
            
        elif self.ui.minfrequency_slider.value() ==1:
            self.fmin=self.sampling_frequency/20
            self.gain_fn(self.Data_ch,p=0,start=9,end=10)

        elif self.ui.minfrequency_slider.value() ==2:
            self.fmin=2*self.sampling_frequency/20
            self.gain_fn(self.Data_ch,p=0,start=8,end=10)

            
        elif self.ui.minfrequency_slider.value() ==3:
            self.fmin=3*self.sampling_frequency/20
            self.gain_fn(self.Data_ch,p=0,start=7,end=10)
  
        datafourier_ch=np.multiply(self.Data_ch,np.exp(1j*self.phase))
        data_ch=np.real(np.fft.ifft(datafourier_ch))
        self.ui.Spectrogram_output.clear()
        self.generate_spectrogram(self.ui.Spectrogram_output,data_ch)
        

        
    def changecolor (self):
        if self.radiobutton_number==3 :
            self.ui.Spectrogram_input.clear()
            self.generate_spectrogram(self.ui.Spectrogram_input,self.amplitude)   
        elif self.radiobutton_number==4  :
            self.ui.Spectrogram_output.clear()
            self.generate_spectrogram(self.ui.Spectrogram_output,self.data_equalized) 
        
        

    
    #spectrogram function:
    def generate_spectrogram(self,widget,data):
        
        fs=self.sampling_frequency
        frequency, time, spectrogram_data = signal.spectrogram(data, fs)
        pg.setConfigOptions(imageAxisOrder='row-major')
        plot =widget.addPlot()
        # Item for displaying image data
        img = pg.ImageItem()
        plot.addItem(img)
        #Add a histogram with which to control the gradient of the image
        hist =pg.HistogramLUTItem()
        # Link the histogram to the image
        hist.setImageItem(img)
        #widget.addItem(hist)
        # Fit the min and max levels of the histogram to the data available
        hist.setLevels(np.min(spectrogram_data), np.max(spectrogram_data))
        hist.gradient.restoreState(self.colorpallet())
        
        img.setImage(spectrogram_data)
        # Scale the X and Y Axis to time and frequency (standard is pixels)
        img.scale(time[-1] / np.size(spectrogram_data, axis=1), frequency[-1] / np.size(spectrogram_data, axis=0))
        # Limit panning/zooming to the spectrogram
        plot.setLimits(xMin=0, xMax=time[-1], yMin=self.fmin, yMax=self.fmax)
        # Add labels to the axis
        plot.setLabel('bottom', "Time", units='s')
        plot.setLabel('left', "Frequency", units='Hz')

    def colorpallet(self):
        self.Gradients = OrderedDict([
            ('thermal', {'ticks': [(0.3333, (185, 0, 0, 255)), (0.6666, (255, 220, 0, 255)), (
                1, (255, 255, 255, 255)), (0, (0, 0, 0, 255))], 'mode': 'rgb'}),
            ('flame', {'ticks': [(0.2, (7, 0, 220, 255)), (0.5, (236, 0, 134, 255)), (0.8, (
                246, 246, 0, 255)), (1.0, (255, 255, 255, 255)), (0.0, (0, 0, 0, 255))], 'mode': 'rgb'}),
            ('yellowy', {'ticks': [(0.0, (0, 0, 0, 255)), (0.2328863796753704, (32, 0, 129, 255)), (0.8362738179251941, (
                255, 255, 0, 255)), (0.5257586450247, (115, 15, 255, 255)), (1.0, (255, 255, 255, 255))], 'mode': 'rgb'}),
            ('bipolar', {'ticks': [(0.0, (0, 255, 255, 255)), (1.0, (255, 255, 0, 255)), (0.5, (
                0, 0, 0, 255)), (0.25, (0, 0, 255, 255)), (0.75, (255, 0, 0, 255))], 'mode': 'rgb'}),
            ('spectrum', {
            'ticks': [(1.0, (255, 0, 255, 255)), (0.0, (255, 0, 0, 255))], 'mode': 'hsv'}),
            ('cyclic', {'ticks': [(0.0, (255, 0, 4, 255)),
                                (1.0, (255, 0, 0, 255))], 'mode': 'hsv'}),
            ('greyclip', {'ticks': [(0.0, (0, 0, 0, 255)), (0.99, (255,
                                                                255, 255, 255)), (1.0, (255, 0, 0, 255))], 'mode': 'rgb'}),
            ('grey', {'ticks': [(0.0, (0, 0, 0, 255)),
                                (1.0, (255, 255, 255, 255))], 'mode': 'rgb'}),
        ])
        self.text = self.ui.comboBox.currentText()
        return self.Gradients[self.text]
    
    def onclicked(self):
        radiobutton=self.sender() # return  toggled radio button 
        if radiobutton.isChecked():
            self.radiobutton_number=radiobutton.value
    def clear(self) :
        if self.radiobutton_number==1 :
            self.ui.input_graph.clear()
        elif self.radiobutton_number==2  :
            self.ui.output_graph.clear()  
        elif self.radiobutton_number==3  :
            self.ui.Spectrogram_input.clear()
        elif self.radiobutton_number==4  :
            self.ui.Spectrogram_output.clear()    

            
    ## pause function
    def pause_fn (self) :
        if self.radiobutton_number==1 :
            sd.stop()
        elif self.radiobutton_number==2  :
            sd.stop()

           
    ## play function :
    def play_fn(self) :
        if self.radiobutton_number==1 :
            sd.play(self.data,self.sampling_frequency)
        elif self.radiobutton_number==2  :
            sd.play(self.data_equalized,self.sampling_frequency)

    ## Scroll Functions  
    def scrollx(self):
        
        position = 0.09 * self.ui.scroll_x.value()
       
        if self.radiobutton_number==1 :
            self.ui.input_graph.plotItem.setXRange(position,(position+0.05))
            self.ui.scroll_x.setMaximum(int(max(self.time)/.09))

        
        elif self.radiobutton_number==2 :
            self.ui.output_graph.plotItem.setXRange(position,(position+0.05))
            self.ui.scroll_x.setMaximum(int(max(self.time)/.09))
            

    
    def zoom (self,x,y):
        if self.radiobutton_number==1 :
            self.ui.input_graph.plotItem.getViewBox().scaleBy((x, y))
            
        elif self.radiobutton_number==2  :
            self.ui.output_graph.plotItem.getViewBox().scaleBy((x, y))

    
    ## pdf function
    def printPDF(self):
        
        self.WIDTH = 210
        pdf=FPDF()
        try:
            shutil.rmtree('plots')
            os.mkdir('plots')
        except FileNotFoundError:
            os.mkdir('plots')
        if self.time.any():
            
            exporter = pg.exporters.ImageExporter(self.ui.input_graph.plotItem)
            exporter.parameters()['width'] = self.ui.input_graph.plotItem.width()
            exporter.export('plots/plot-1.png')
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(40 ,10 , 'original signal')
            pdf.image('plots/plot-1.png', 15, 30, self.WIDTH - 30)
            exporter = pg.exporters.ImageExporter(self.ui.Spectrogram_input.scene())
            exporter.export('plots/spec-1.png')
            if self.ui.input_graph.plotItem.height()>600 or self.ui.Spectrogram_input.height()>600 :
                pdf.add_page()
                pdf.image('plots/spec-1.png' , 15, 30, self.WIDTH - 30)
            else:
                pdf.image('plots/spec-1.png' , 15, self.WIDTH / 2 +50 , self.WIDTH - 30)
        
        if self.data_equalized.any():
            exporter = pg.exporters.ImageExporter(self.ui.output_graph.plotItem)
            exporter.parameters()['width'] = self.ui.output_graph.plotItem.width()
            exporter.export('plots/plot-2.png')
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(40 ,10 , 'Modified signal')
            pdf.image('plots/plot-2.png', 15, 30, self.WIDTH - 30)
            
            exporter = pg.exporters.ImageExporter(self.ui.Spectrogram_output.scene())
            exporter.export('plots/spec-2.png')
            if self.ui.output_graph.plotItem.height()>600 or self.ui.Spectrogram_output.height()>600 :
                pdf.add_page()
                pdf.image('plots/spec-2.png'  , 15, 30, self.WIDTH - 30)
            else:
                pdf.image('plots/spec-2.png'  , 15, self.WIDTH / 2 + 50, self.WIDTH - 30)
    
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export PDF", None, "PDF files (.pdf)")
        if fn:
            if QtCore.QFileInfo(fn).suffix() == "":
                fn += ".pdf"   
        
        pdf.output(f'{fn}', 'F')
        try:
            shutil.rmtree('plots')
        except:
            pass

    def new_window (self):
        self.newwindow=ApplicationWindow()
        self.newwindow.show()
             
    

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__))) # to load the directory folder

    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()




if __name__ == "__main__":
    main()


    
