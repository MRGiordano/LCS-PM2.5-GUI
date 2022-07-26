# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:17:31 2021

@author: Mike
"""

import pandas as pd
import numpy as np
import scipy as sp

import matplotlib
import matplotlib.pyplot as plt

from statsmodels.formula.api import glm

import tkinter as tk
from tkinter import filedialog, ttk, E, W, N, S, END
from tkinter.messagebox import showinfo, showwarning
from tkinter.scrolledtext import ScrolledText

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.transforms as transform

from sklearn.model_selection import train_test_split
from sklearn import metrics

#initialize figures
width, height = plt.figaspect(0.5)
fig = Figure(figsize=(width, height), dpi=100)


class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        
        #initialize properties
        self.__refdata = pd.DataFrame()
        self.__lcsdata = pd.DataFrame()
        self.__pmbound_hold = tk.DoubleVar()
        self.__rhbound_hold = tk.DoubleVar()
        self.__tbound_hold = tk.DoubleVar()
        self.__intbound_hold = tk.DoubleVar()
        self.__multifitframe_var = tk.IntVar()
        self.__pmbound2_hold = tk.DoubleVar()
        self.__rhbound2_hold = tk.DoubleVar()
        self.__tbound2_hold = tk.DoubleVar()
        self.__intbound2_hold = tk.DoubleVar()
        self.__regridoption = tk.StringVar()
        self.__resamplebasis = tk.DoubleVar()
        self.__cleanlcs0 = tk.BooleanVar()
        self.__cleanref0 = tk.BooleanVar()
        self.__filterref = tk.BooleanVar()
        self.__filterlcs = tk.BooleanVar()
        self.__showglmresults = tk.BooleanVar()
        
        self.__pmcoeff = tk.DoubleVar()
        self.__tcoeff = tk.DoubleVar()
        self.__rhcoeff = tk.DoubleVar()
        self.__intcoeff = tk.DoubleVar()
        
        self.__pmcoeff2 = tk.DoubleVar()
        self.__tcoeff2 = tk.DoubleVar()
        self.__rhcoeff2 = tk.DoubleVar()
        self.__intcoeff2 = tk.DoubleVar()
        
        self.title('LCS Linear Fit GUI')
        self.geometry('960x800')
        self.resizable(1,1)
        
        #tabs
        names = ['Main','Options']
        self.nb = self.create_notebook(names)
        
        
        #frames
        self.nb.tabs['Main'].grid_propagate(0)
        for x in range(4):
            self.nb.tabs['Main'].grid_rowconfigure(x, weight=1)
        for y in range(4):
            self.nb.tabs['Main'].grid_columnconfigure(y, weight=0)
        
        
        self.refdata_button = tk.Button(self.nb.tabs['Main'],text='Load Reference Data', width=20, height = 10, command=(lambda: self.open_file('ref')))
        self.refdata_button.grid(row=0,column=0,sticky=E+W)

        self.lcsdata_button = tk.Button(self.nb.tabs['Main'], text='Load LCS Data', width=20, height = 10, command=(lambda: self.open_file('lcs')))
        self.lcsdata_button.grid(row=1,column=0,sticky=E+W)

        self.comp_button = tk.Button(self.nb.tabs['Main'], text='Compare and Fit', width=20, height = 10, command=(lambda: self.do_comp()))
        self.comp_button.grid(row=2,column=0,sticky=E+W)
        
        #canvas
        self.canvas = FigureCanvasTkAgg(fig,self.nb.tabs['Main'])
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0,column=1,columnspan=3,rowspan=3,sticky=N+S+E+W)
        
        self.txtbox = ScrolledText(self.nb.tabs['Main'],height=10)
        self.txtbox.grid(row=3,column=0,columnspan=4,sticky=E+W)
        

        self.options_tab_draw()

        
    def options_tab_draw(self):
        #frame for data options
        self.data_frame = tk.Frame(self.nb.tabs['Options'],borderwidth=1,highlightthickness=1,highlightbackground="black")
        self.data_frame.grid(row=0,column=0,columnspan=3,padx=10,pady=10,sticky=E+W)
        tk.Label(self.data_frame,text='Data Options').grid(row=0,column=0,columnspan=3,sticky=E+W)
        
        #labels and entrys in data options frame
        tk.Label(self.data_frame,text='Date format').grid(row=1,column=0)
        self.dateformat = tk.Entry(self.data_frame,width=20)
        self.dateformat.insert(END, 'not yet implemented')
        self.dateformat.config(state='disabled')
        self.dateformat.grid(row=1,column=1,columnspan=3,sticky=W+E)
        
        tk.Label(self.data_frame,text='Regrid Data').grid(row=2,column=0)
        self.regriddata_val = tk.Entry(self.data_frame,width=8,validate='key',vcmd=(self.data_frame.register(validate_float),'%P'))
        self.regriddata_val.grid(row=2,column=1,sticky=W+E)
        rgrd_options = ['Minutes','Hours','Days']
        self.regriddata_option = tk.OptionMenu(self.data_frame,self.__regridoption,*rgrd_options)
        self.regriddata_option.grid(row=2,column=2,columnspan=2,sticky=W+E)
        
        tk.Label(self.data_frame,text='Test/train split ratio').grid(row=3,column=0)
        self.ttsplitratio = tk.Entry(self.data_frame,width=4,validate='key',vcmd=(self.data_frame.register(validate_float),'%P'))
        self.ttsplitratio.insert(0,'0.2')
        self.ttsplitratio.grid(row=3,column=1,sticky=W+E)
        
        tk.Label(self.data_frame,text='Remove \'0\'s from data').grid(row=4,column=0)
        self.cleanref0s = tk.Checkbutton(self.data_frame,text='Reference', variable=self.__cleanref0)
        self.cleanref0s.grid(row=4,column=1,sticky=W+E)
        self.cleanlcs0s = tk.Checkbutton(self.data_frame,text='LCS', variable=self.__cleanlcs0)
        self.cleanlcs0s.grid(row=4,column=2,sticky=W+E)
        
        tk.Label(self.data_frame,text='High pass filter').grid(row=5,column=0)
        self.hipassfilter = tk.Entry(self.data_frame,width=8,validate='key',vcmd=(self.data_frame.register(validate_float),'%P'))
        self.hipassfilter.grid(row=5,column=1,sticky=W+E)
        self.filterref = tk.Checkbutton(self.data_frame,text='Reference', variable=self.__filterref)
        self.filterref.grid(row=5,column=2,sticky=W)
        self.filterlcs = tk.Checkbutton(self.data_frame,text='LCS', variable=self.__filterlcs)
        self.filterlcs.grid(row=5,column=3)
        
        #frame for coefficient options
        self.coeffs_frame = tk.Frame(self.nb.tabs['Options'],width=100,height=50,borderwidth=1,highlightthickness=1,highlightbackground="black")
        self.coeffs_frame.grid(row=1,column=0,padx=10,pady=10,sticky=W)
        tk.Label(self.coeffs_frame,text='Coefficient Options').grid(row=0,column=0,columnspan=2,sticky=N+S+W+E)
        
        #labels, entrys, and checkboxes for coefficient options frame
        tk.Label(self.coeffs_frame,text='Set PM Coefficient').grid(row=1,column=0)
        self.pmbound = tk.Entry(self.coeffs_frame,width=8,validate='key',vcmd=(self.coeffs_frame.register(validate_float),'%P'))
        self.pmbound.grid(row=1,column=1)
        
        tk.Label(self.coeffs_frame,text='Set RH Coefficient').grid(row=2,column=0)
        self.rhbound = tk.Entry(self.coeffs_frame,width=8,validate='key',vcmd=(self.coeffs_frame.register(validate_float),'%P'))
        self.rhbound.grid(row=2,column=1)
        
        tk.Label(self.coeffs_frame,text='Set T Coefficient').grid(row=3,column=0)
        self.tbound = tk.Entry(self.coeffs_frame,width=8,validate='key',vcmd=(self.coeffs_frame.register(validate_float),'%P'))
        self.tbound.grid(row=3,column=1)
        
        tk.Label(self.coeffs_frame,text='Set Intercept to 0').grid(row=4,column=0)
        self.intbound_hold_button = tk.Checkbutton(self.coeffs_frame,text='',variable = self.__intbound_hold)
        self.intbound_hold_button.grid(row=4,column=1)
        
        #frame for multifit
        self.multifitframe = tk.Frame(self.nb.tabs['Options'],highlightthickness=1,highlightbackground="black")
        
        #checkbox for multifit
        self.multifit_button = tk.Checkbutton(self.nb.tabs['Options'],text='Multifit',variable = self.__multifitframe_var,command=self.multifitframe_disp)
        self.multifit_button.grid(row=2,column=0,columnspan=3,padx=10,pady=0,sticky=W)
        
        #multifit frame entrys
        tk.Label(self.multifitframe,text='Break value').grid(row=1,column=0,sticky=W)
        self.multifit_break = tk.Entry(self.multifitframe,width=8,validate='key',vcmd=(self.multifitframe.register(validate_float),'%P'))
        self.multifit_break.grid(row=1,column=1,stick=W)
        
        tk.Label(self.multifitframe,text='Set PM Coefficient').grid(row=2,column=0)
        self.pmbound2 = tk.Entry(self.multifitframe,width=8,validate='key',vcmd=(self.multifitframe.register(validate_float),'%P'))
        self.pmbound2.grid(row=2,column=1)
        
        tk.Label(self.multifitframe,text='Set RH Coefficient').grid(row=3,column=0)
        self.rhbound2 = tk.Entry(self.multifitframe,width=8,validate='key',vcmd=(self.multifitframe.register(validate_float),'%P'))
        self.rhbound2.grid(row=3,column=1)
        
        tk.Label(self.multifitframe,text='Set T Coefficient').grid(row=4,column=0)
        self.tbound2 = tk.Entry(self.multifitframe,width=8,validate='key',vcmd=(self.multifitframe.register(validate_float),'%P'))
        self.tbound2.grid(row=4,column=1)
        
        tk.Label(self.multifitframe,text='Set Intercept to 0').grid(row=5,column=0)
        self.intbound_hold_button2 = tk.Checkbutton(self.multifitframe,text='',variable = self.__intbound2_hold)
        self.intbound_hold_button2.grid(row=5,column=1)
        
        #frame for misc
        self.miscframe = tk.Frame(self.nb.tabs['Options'], highlightthickness=1, highlightbackground='black')
        self.miscframe.grid(row=0,column=3,padx=10,pady=10)
        
        #stuff inside misc frame
        tk.Label(self.miscframe, text='Show full GLM fit summary').grid(row=0,column=0,padx=10,pady=10)
        self.showglmresults = tk.Checkbutton(self.miscframe,text='',variable=self.__showglmresults)
        self.showglmresults.grid(row=0,column=1)
        
        self.save_button = tk.Button(self.miscframe, text='Save csv with calibrated data', width=10, height = 5, state='disabled',command=(lambda: self.save_data()))
        self.save_button.grid(row=1,column=0,columnspan=2,sticky=E+W)
        
    def open_file(self, ref_or_lcs):    
        if ref_or_lcs == 'ref':
            file = filedialog.askopenfilename(initialdir='/',
                                          title='Select a file',
                                          filetype=(('csv files','*.csv'), ("all files", "*.*")))
        elif ref_or_lcs == 'lcs':
            file = filedialog.askopenfilenames(initialdir='/',
                                          title='Select a file',
                                          filetype=(('csv files','*.csv'), ("all files", "*.*")))
        
        if file:
            if ref_or_lcs == 'ref':
                df = pd.read_csv(file,sep=',')
                df.columns = map(str.lower, df.columns)
                df = self.df_check(df,ref_or_lcs)
                
                self.set_refdata(df)
                self.draw_ref_graph(df)
                
            elif ref_or_lcs == 'lcs':
                dflist = []
                for filename in file:
                    df = pd.read_csv(filename, sep=',')
                    df.columns = map(str.lower, df.columns)
                    df = self.df_check(df,ref_or_lcs)
                    dflist.append(df)
                    
                df = pd.concat(dflist,axis=0).groupby(level=0).median()
                self.set_lcsdata(df)
                self.draw_lcs_graph(df)
            
            txtstr = 'Loaded file: %s as %s data.\r' %(file,ref_or_lcs)
            self.txtbox.insert(END,txtstr+'\n')
    
    #tab method
    def create_notebook(self,names):
        nb = MyNotebook(self,names)
        nb.pack(expand=True,fill='both')
        
        def add_label(parent,text,row,column):
            label = tk.Label(parent,text=text)
            label.grid(row=row,column=column, sticky=tk.N,pady=10)
            return label
        
        return nb
        
    def draw_ref_graph(self,df):
        plot1 = fig.add_subplot(3,3,(1,3))
        if plot1.has_data():
            plot1.clear()
        y = df['data'].values
        x = df.index
        plot1.plot(x,y,marker='o',color='black')
        plot1.set_ylabel('Reference Data')
        M=5
        xticks = matplotlib.ticker.MaxNLocator(M)
        plot1.xaxis.set_major_locator(xticks)
        self.canvas.draw_idle()

    def draw_lcs_graph(self,df):
        plot2 = fig.add_subplot(3,3,(4,6))
        if plot2.has_data():
            plot2.clear()
        y = df['data'].values
        x = df.index
        plot2.plot(x,y,marker='o',color='r')
        plot2.set_ylabel('LCS Data')
        M=5
        xticks = matplotlib.ticker.MaxNLocator(M)
        plot2.xaxis.set_major_locator(xticks)
        self.canvas.draw()
    
    def draw_comp_graph(self,x,y,rh,multiplot):
        ax = fig.add_subplot(3,3,7)
        if not multiplot:
            # if ax.has_data():
            ax.clear()
            cbar = fig.colorbar(ax.scatter(x,y[:,0], c=rh, cmap='jet', marker='.'), orientation='vertical')
            cbar.remove()
        ax.scatter(x,y[:,0], c=rh, cmap='jet', marker='.')
        axislim_l = min(min(x),min(y[:,0])) * 0.95
        axislim_u = max(max(x),max(y[:,0])) * 1.05
        ax.axis(ymin = axislim_l, ymax = axislim_u, xmin=axislim_l, xmax=axislim_u)
        ax.plot([0, 1], [0, 1], color='black', transform=ax.transAxes)
        ax.set_xlabel('Reference Training Set')
        ax.set_ylabel('LCS Training Set')
        
        if not multiplot:
            cbar = fig.colorbar(ax.scatter(x,y[:,0], c=rh, cmap='jet', marker='.'), orientation='vertical')
            cbar.set_label('RH')
        else:
            cbar = fig.colorbar(ax.scatter(x,y[:,0], c=rh, cmap='Greys', marker='.'), orientation='vertical')
            cbar.set_label('RH')

        fig.tight_layout(pad=3.0)
        self.canvas.draw()
        
    def draw_fit_graph(self,x,y_ref,y_pred,multiplot):
        ax = fig.add_subplot(3,3,(8,9))
        if not multiplot:
            if ax.has_data():
                ax.clear()
        
        ax.plot(x,y_ref,color='black')
        ax.scatter(x,y_pred,marker='.',color='red')
        
        M=5
        xticks = matplotlib.ticker.MaxNLocator(M)
        ax.xaxis.set_major_locator(xticks)
        
        self.canvas.draw()

    def do_comp(self):
        print (self.__filterlcs.get())
        figaxes = fig.get_axes()
        if len(figaxes) > 2:
            if figaxes[2].has_data():
                figaxes[2].clear()
            if len(figaxes) == 4:
                if figaxes[3].has_data():
                    figaxes[3].clear()
        
        df_lcs = self.__lcsdata
        df_ref = self.__refdata
        
        if self.__cleanlcs0:
            df_lcs = clean0(df_lcs)    
        if self.__cleanref0:
            df_ref = clean0(df_ref)
        
        if self.__filterlcs.get():
            df_lcs = hipassfilter(df_lcs, int(self.hipassfilter.get()), 'data')
        if self.__filterref.get():
            df_ref = hipassfilter(df_ref, int(self.hipassfilter.get()), 'data')
            
        if len(self.regriddata_val.get()) > 0:
            strng = str(self.regriddata_val.get())
            if self.__regridoption.get() == 'Minutes':
                strng += 'min'
            elif self.__regridoption.get() == 'Hours':
                strng += 'H'
            elif self.__regridoption.get() == 'Days':
                strng += 'D'
            elif not self.__regridoption.get():
                tk.messagebox.showerror(self,'Select a resample basis or remove characters from resample box') 
                return None
            
            df_lcs = df_lcs.resample(rule=strng).mean()
            df_ref = df_ref.resample(rule=strng).mean()
            
        comp_df = pd.DataFrame()
        comp_df = pd.merge(left=df_ref,right=df_lcs,how='outer',left_index=True,right_index=True)
        comp_df = comp_df.rename(columns={'data_x':'ref','data_y':'lcs'})
        comp_df.dropna(axis=0,inplace=True)
        
        if self.__multifitframe_var.get():
            #split df and make arrays
            comp_df_lower = comp_df.loc[comp_df['lcs'] <= float(self.multifit_break.get())]
            comp_df_upper = comp_df.loc[comp_df['lcs'] > float(self.multifit_break.get())]
            
            X_lower = comp_df_lower[['lcs','t','rh']].values
            X_upper = comp_df_upper[['lcs','t','rh']].values
            
            y_lower = comp_df_lower[['ref']].values
            y_upper = comp_df_upper[['ref']].values
            
            #lower first
            X_train, X_test, y_train, y_test = testtrainsplit(X_lower, y_lower, test_size=float(self.ttsplitratio.get()), randomstate=0)
            
            if self.__intbound_hold.get():
                formula = 'y ~ lcs + rh + t -1'
            else:
                formula = 'y ~ lcs + rh + t'
            
            formula_df =  pd.DataFrame({'y' : y_train[:,0],'lcs':X_train[:,0],'t':X_train[:,1],'rh':X_train[:,2]})
            glmm = glm(formula=formula,data=formula_df)
            
            if len(self.checkfithold(1)) != 0:
                glmm_results_lower = glmm.fit_constrained(constraints=self.checkfithold(1))
            else:
                glmm_results_lower = glmm.fit()
            
            self.txtbox.insert(END, '==============================================\n')
            self.txtbox.insert(END, 'Lower Bound fit set\n')
            self.txtbox.insert(END, '----------------------------------------------\n')

            self.printglmresults(glmm_results_lower.params,glmm_results_lower.bse)
            y_test_lower = y_test
            
            test_df = pd.DataFrame({'lcs':X_test[:,0],'t':X_test[:,1],'rh':X_test[:,2]})
            y_pred_test_lower = glmm_results_lower.predict(test_df)
            self.getfitstats(y_test_lower,y_pred_test_lower)
            
            self.set_globalvoeffs(glmm_results_lower.params,0)
            self.draw_comp_graph(y_train, X_train, formula_df['rh'],False)
            
            y_pred_all_lower = glmm_results_lower.predict(comp_df_lower)
            
            #upper
            X_train, X_test, y_train, y_test = testtrainsplit(X_upper, y_upper, test_size=float(self.ttsplitratio.get()), randomstate=0)
            
            if self.__intbound2_hold.get():
                formula = 'y ~ lcs + rh + t -1'
            else:
                formula = 'y ~ lcs + rh + t'
            
            formula_df =  pd.DataFrame({'y' : y_train[:,0],'lcs':X_train[:,0],'t':X_train[:,1],'rh':X_train[:,2]})
            glmm = glm(formula=formula,data=formula_df)
            
            if len(self.checkfithold(2)) != 0:
                glmm_results_upper = glmm.fit_constrained(constraints=self.checkfithold(2))
            else:
                glmm_results_upper = glmm.fit()
            
            self.txtbox.insert(END, '==============================================\n')
            self.txtbox.insert(END, 'Upper Bound fit set\n')
            self.txtbox.insert(END, '----------------------------------------------\n')
            
            self.printglmresults(glmm_results_upper.params,glmm_results_upper.bse)
            y_test_upper = y_test
            
            test_df = pd.DataFrame({'lcs':X_test[:,0],'t':X_test[:,1],'rh':X_test[:,2]})
            y_pred_test_upper = glmm_results_upper.predict(test_df)
            self.getfitstats(y_test_upper,y_pred_test_upper)
            
            self.draw_comp_graph(y_train, X_train, formula_df['rh'],True)
            
            y_pred_all_upper = glmm_results_upper.predict(comp_df_upper)
            
            #recombine upper and lower and plot
            comp_df_lower['pred'] = y_pred_all_lower
            comp_df_upper['pred'] = y_pred_all_upper
            
            combodf = pd.concat([comp_df_lower,comp_df_upper])
            combodf = combodf.sort_index()
            
            self.set_globalvoeffs(glmm_results_upper.params,1)
            self.draw_fit_graph(combodf.index, combodf['ref'], combodf['pred'], False)
            self.save_button['state'] = 'normal'
            
        else:
            X = comp_df[['lcs','t','rh']].values
            y = comp_df[['ref']].values
            
            X_train, X_test, y_train, y_test = testtrainsplit(X, y, test_size=float(self.ttsplitratio.get()), randomstate=0)
        
            if self.__intbound_hold.get():
                formula = 'y ~ lcs + rh + t -1'
            else:
                formula = 'y ~ lcs + rh + t'
            
            formula_df =  pd.DataFrame({'y' : y_train[:,0],'lcs':X_train[:,0],'t':X_train[:,1],'rh':X_train[:,2]})
            glmm = glm(formula=formula,data=formula_df)
            
            if len(self.checkfithold(1)) != 0:
                glmm_results = glmm.fit_constrained(constraints=self.checkfithold(1))
            else:
                glmm_results = glmm.fit()
            
            test_df = pd.DataFrame({'lcs':X_test[:,0],'t':X_test[:,1],'rh':X_test[:,2]})
            y_pred_test = glmm_results.predict(test_df)
            
            
            y_pred_all = glmm_results.predict(comp_df)
            
            self.set_globalvoeffs(glmm_results.params,0)
            self.printglmresults(glmm_results.params,glmm_results.bse)
            self.getfitstats(y_test,y_pred_test)
            self.draw_comp_graph(y_train, X_train, formula_df['rh'],False)
            self.draw_fit_graph(comp_df.index, comp_df['ref'], y_pred_all, False)
     
        
        if self.__showglmresults.get():
            self.txtbox.insert(END, glmm_results.summary())
        
        self.save_button['state'] = 'normal'
    
    def set_globalvoeffs(self,coeffsarray,multifit):
        if multifit == 0:
            self.__intcoeff.set(coeffsarray[0])
            self.__pmcoeff.set(coeffsarray[1])
            self.__rhcoeff.set(coeffsarray[2])
            self.__tcoeff.set(coeffsarray[3])
        
        elif multifit == 1:
            self.__intcoeff2.set(coeffsarray[0])
            self.__pmcoeff2.set(coeffsarray[1])
            self.__rhcoeff2.set(coeffsarray[2])
            self.__tcoeff2.set(coeffsarray[3])
        
    
    def save_data(self):
        if self.__multifitframe_var.get():
            return True
            #not yet implemented
        
        else:
            file = filedialog.askopenfilenames(initialdir='/',
                                          title='Select a file',
                                          filetype=(('csv files','*.csv'), ("all files", "*.*")))
            if file:
                dflist = []
                for filename in file:
                    df = pd.read_csv(filename, sep=',')
                    df.columns = map(str.lower, df.columns)
                    df = self.df_check(df,'lcs')
                    dflist.append(df)
                
                i=0
                while i < len(dflist):
                    dflist[i]['calibrated'] = (dflist[i]['data']*self.__pmcoeff.get()) + (dflist[i]['rh']*self.__rhcoeff.get()) + (dflist[i]['t']*self.__tcoeff.get()) + self.__intcoeff.get()
                    dflist[i].to_csv(path_or_buf = file[i])
                    i+=1
                
                
            
        
        
    def df_check(self,df,ref_or_lcs):
        
        #check all columns are there
        if 'time' and 'data' not in df.columns:
            tk.messagebox.showerror(self,'time or data columns not found in loaded file')
            return None
        if ref_or_lcs == 'lcs':
            # if not (df.columns.str.contains('rh')).any() or not (df.columns.str.contains(r'^t[^i]')).any():
            if 'rh' and 't' not in df.columns:
                tk.messagebox.showerror(self,'RH and/or T columns not found in loaded file') 
                return None
                
        #change time to datetime and tz-naive and set index
        df['time'] = pd.to_datetime(df['time'])
        if df['time'][0].tzinfo is not None:
            df = df.set_index('time')
            df = df.tz_localize(None)
        else:
            df = df.set_index('time')
        
        return df
    
    def df_cleanup(self,df,*colofinterest,**kwargs):
        if 'remove0' in kwargs.values():
            for col in colofinterest.values():
                newdf = df.drop(df[colofinterest] == 0)
        
        if 'removehighs' in kwargs.values():
            for col in colofinterest.values():
                newdf = df.drop(df[colofinterest] > self.hipassfilter.get())

        return newdf
    
    def getfitstats(self,y_ref, y_pred):
        if y_ref.ndim > 1:
            y_ref = y_ref.reshape(-1)
        if y_pred.ndim > 1:
            y_pred = y_pred.reshape(-1)
        
        mnb = np.sum(y_pred-y_ref) / np.sum(y_ref)
        nbias = (1/len(y_ref))*np.sum(y_pred - y_ref)
        
        CvMAE = np.sum(abs(y_pred - nbias - y_ref)) / np.sum(y_ref)
        
        mae =  metrics.mean_absolute_error(y_ref, y_pred)
        mse =  metrics.mean_squared_error(y_ref, y_pred)
        rmse =  np.sqrt(metrics.mean_squared_error(y_ref, y_pred))
        r =  sp.stats.pearsonr(y_ref, y_pred)
        
        self.txtbox.insert(END,'--------------------------------------\n')  
        self.txtbox.insert(END,'Fit stats for testing data: \n')
        self.txtbox.insert(END,'Mean normalized bias: %.3f\n' %mnb)  
        self.txtbox.insert(END,'Mean Absolute Error: %.3f\n' %mae)
        self.txtbox.insert(END,'Mean Squared Error: %.3f\n' %mse)  
        self.txtbox.insert(END,'Root Mean Squared Error: %.3f\n' %rmse)  
        self.txtbox.insert(END,'Pearson r: %.3f\n' %r[0])  
        self.txtbox.insert(END,'CvMAE: %.3f\n' %CvMAE)  
    
        
    def printglmresults(self,params,errors):
        self.txtbox.insert(END, '-------------------------\n')
        self.txtbox.insert(END,'Coefficients for fitting a Generalized Linear Model to the training data:\n')
        for index in params.index:
            strng = index.capitalize() + ': ' + '%.3f' %params[index] + ', std err: ' + '%.3f' %errors[index]+'\n'
            self.txtbox.insert(END, strng)
        if self.__intbound_hold.get() == 1 or self.__intbound2_hold.get() == 1:
            strng = 'Overall formula: PM = '+ '%.3f'%params['lcs']+'*LCS' + ' + %.3f'%params['t']+'*T' + ' + %.3f'%params['rh']+'*RH' +'\n'
        else:
            strng = 'Overall formula: PM = '+ '%.3f'%params['lcs']+'*LCS' + '%.3f'%params['t']+'*T' + '%.3f'%params['rh']+'*RH' +' + ' '%.3f'%params['Intercept']+'\n'
        self.txtbox.insert(END,strng)
            
        
    def multifitframe_disp(self):
        if self.__multifitframe_var.get():
            self.multifitframe.grid(row=3,column=0,padx=10,pady=10,sticky=W)
        else:
            self.multifitframe.grid_forget()
    
    def checkfithold(self,lu):
        constraintstr = ''
        
        #normal constraints set
        if lu == 1:
            if len(self.pmbound.get()) != 0:
                constraintstr += 'lcs = ' + self.pmbound.get()
            if len(self.rhbound.get()) != 0:
                if len(constraintstr) !=0:
                    constraintstr += ','
                constraintstr += 'rh = ' + self.rhbound.get()
            if len(self.tbound.get()) != 0:
                if len(constraintstr) !=0:
                    constraintstr += ','
                constraintstr += 't = '  + self.tbound.get()
        #multifit constraints set
        elif lu == 2:
            if len(self.pmbound2.get()) != 0:
                constraintstr += 'lcs = ' + self.pmbound2.get()
            if len(self.rhbound2.get()) != 0:
                if len(constraintstr) !=0:
                    constraintstr += ','
                constraintstr += 'rh = ' + self.rhbound2.get()
            if len(self.tbound2.get()) != 0:
                if len(constraintstr) !=0:
                    constraintstr += ','
                constraintstr += 't = '  + self.tbound2.get()

        return constraintstr
        
        
    #get and set functions
    def get_refdata(self):
        return self.__refdata
    
    def set_refdata(self, x):
        self.__refdata = x
    
    def get_lcsdata(self):
        return self.__lcsdata
    
    def set_lcsdata(self, x):
        self.__lcsdata = x
    
    def get_pmbound_hold(self):
        return self.__pmbound_hold
    
    def set_pmbound_hold(self,x):
        self.__pmbound_hold = x


class MyNotebook(ttk.Notebook):
    def __init__(self,master,names):
        super().__init__(master,width=800,height=800)
        
        self.tabs={}
        for name in names:
            self.tabs[name] = tab = ttk.Frame(self)
            self.add(tab,text=name)

def testtrainsplit(X,y,test_size,randomstate=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=randomstate)
    return X_train, X_test, y_train, y_test

def clean0(df):
    df.loc[~(df==0).all(axis=1)]
    
    return df

def hipassfilter(df, val, col):
    df = df.loc[~(df[col] > val)]
    
    return df

def validate_float(i):
    try:
        float(i)
    except:
        return False
    return True
 

if __name__ == "__main__":

    gui = GUI()
    
    gui.mainloop()
