from pyexcel.cookbook import merge_all_to_a_book
import glob
import xlrd
from prettytable import PrettyTable

merge_all_to_a_book(glob.glob("profile_summary.csv"), "profile_summary.xls")
loc = ("profile_summary.xls")
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)
pcie_hd =sheet.cell_value(55, 7) #H56
pcie_dh =sheet.cell_value(56, 7) #H57
kernel_time_avg=1
rdddr=sheet.cell_value(61, 9) #J62
wtddr=sheet.cell_value(62, 9) #J63
knfg=sheet.cell_value(45, 4) #E46

print("=============Average Execution Time Breakdwon per Iteration============")
t = PrettyTable(['PCIE Host->Device', 'PCIE Device->Host','Kernel-Total','Kernel-Compute','Kernel-DDR_READ','Kernel-DDR_WRITE'])
t.add_row([str(pcie_hd)+" ms",str(pcie_dh)+" ms",kernel_time_avg,str(knfg)+" ms",str(rdddr)+" ns",str(wtddr)+" ns"])
print(t)