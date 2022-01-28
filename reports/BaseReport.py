import json

class BaseReport:
    # Base class for all the reports
    def __init__(self,imgstack_file,coord_info):
        
        self.imgstack = np.load(imgstack_file)
        a_file = open(coord_info, "r")
        self.coords = json.load(a_file)

        self.pdf = None
    #PDF controls--------------------------------------------
    def set_pdf(self,pdf)-> None:
        self.pdf = pdf
    
    def get_pdf(self):
        return self.pdf

    def isPdf(self)->bool:
        if self.pdf is None:
            return False
        return True
    
    def closePdf(self)->None:
        if self.isPdf():
            self.pdf.close()
            self.pdf = None
