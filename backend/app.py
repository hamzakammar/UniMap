import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
from flask import Flask, render_template, send_file, send_from_directory, jsonify
from markupsafe import escape
import networkx as nx
from flask_cors import CORS, cross_origin
from PIL import Image
import cv2
import numpy as np
import pytesseract


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

number = 0
@app.route('/')
def hello():
    return jsonify({
        "message": "UniMap API",
        "version": "1.0.0",
        "endpoints": {
            "route": "/<start>/<end> - Get navigation route between two rooms",
            "health": "/health - API health check"
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

G = nx.Graph()
# ERC First Floor (G)
# // Add Nodes

G_poses = [(4.0, 3.0),(2.6, 3.0),(2.6, 3.6),(5.4, 2.6),(4.1, 8.0),(4.1, 8.9),(3.3, 8.3),(3.3, 9.0),(3.4, 12.0),(4.1, 11.9),(4.1, 14),(3.2, 13.0),(6.5, 14),
(8.5, 14),(10.2, 14),(6.1, 12.0),(4.1, 15.2),(11.2, 15.0),(8.9, 15.2),(11.3, 11.3),(7.5, 12.4),(8.5, 12.4),(9.5, 12.4)]
G_names = ["A", "Stair", "Elevator", "1902", "B", "C", "1004", "1003", "1002", "D", "E", "1001", "F", "G", "H", "1006", "1802", "1012", "1012B", "1011", "1007", "1008", "1009"]
# G_Names = ["A", "B", "C", "D", "E", "F", "G", "H", "Stair", "Elevator", "1001", "1002", "1003", "1004", "1006", "1007", "1009", "1011", "1012", "1012B", "1802", "1902"]
# G_Poses = [(3,4), (4,7.5), (4,8), (4, 10.5), (4, 11.5), (6, 11.5), (8, 11.5), (10, 11.5), (2.5, 3.5), (2.5, 4), (3, 11), (3,10), (3,8), (3,7.5), (6, 10.5), (7, 10.5), (9, 10), (10,9), (10, 13), (8, 13), (4, 13.5), (6, 3.5) ]

for i in range(len(G_poses)):
    G.add_node(G_names[i], pos = G_poses[i])
print(G.nodes)
G.add_node("Cali", pos = (0,0))
G.add_node("Cali2", pos = (12,15))
# // Add edges
G.add_edge("A", "B")
G.add_edge("A", "Stair")
G.add_edge("A", "Elevator")
G.add_edge("A", "1902")
G.add_edge("B", "1004")
G.add_edge("B", "C")
G.add_edge("C", "1003")
G.add_edge("C", "D")
G.add_edge("D", "1002")
G.add_edge("D", "E")
G.add_edge("D", "1001")
G.add_edge("E", "F")
G.add_edge("E", "1001")
G.add_edge("E", "1802")
G.add_edge("F", "G")
G.add_edge("G", "H")
G.add_edge("G", "1009")
G.add_edge("F", "1007")
G.add_edge("F", "1006")
G.add_edge("H", "1012")
G.add_edge("1012", "1012B")
G.add_edge("H", "1011")
G.add_edge("G", "1008")
# print(G.edges())

# #ERC second floor (I)

I = nx.Graph()
I_names = ["A", "Stair", "Elevator", "2902", "2906", "B","2001", "2002", "C", "2003", "2004",
"D", "2007", "2008", "E", "2006", "2009", "G", "Stair2", "2909", "F", "H", "2011", "2012",
"2013", "2014", "I", "2016", "2017", "2018", "2019", "J", "2022", "2023", "2024", "2026", "K",
"2027", "2028", "2029", "2031", "L", "2907", "M", "2033", "2032"]

I_poses = [(3.7, 2.8),(2.7, 2.8),(2.7, 3.8),(5.2, 2.6),(3.1, 4.5),(3.9, 6.0),(4.4, 6.6),(3.5, 6.1),(4.0, 8.2),(3.5, 7.9),(3.5, 8.5),(3.9, 12.1),(3.5, 11.2),(3.4, 11.9),(9.6, 12.6),
(9.2, 11.8),(10.2, 11.8),(17.4, 12.6),(17.4, 13.7),(15.9, 13.7),(14.0, 12.6),(14.0, 9.0),(13.6, 9.5),(14.3, 9.3),(13.6, 8.7),(14.3, 8.7),(14.0, 6.1),(13.6, 6.5),(14.4, 6.5),(13.6, 5.8),
(14.4, 5.9),(13.9, 3.1),(14.4, 3.7),(14.4, 3.0),(14.0, 2.4),(13.5, 2.4),(11.1, 2.8),(11.5, 3.3),(11.2, 2.3),(10.9, 3.3),(10.6, 2.4),(8.9, 3.1),(8.4, 3.7),(8.7, 5.5),(9.8, 5.5),(8.1, 5.0)]

for i in range(len(I_names)):
    I.add_node(I_names[i], pos = I_poses[i])
print(I.nodes(data=True))
I.add_node("Cali", pos = (0,0))
I.add_node("Cali2", pos = (20,15))
# // Add edges
I.add_edge("A", "Stair")
I.add_edge("A", "Elevator")
I.add_edge("A", "2902")
I.add_edge("A", "2906")
I.add_edge("A", "B")
I.add_edge("B", "2001")
I.add_edge("B", "2002")
I.add_edge("B", "C")
I.add_edge("C", "2003")
I.add_edge("C", "2004")
I.add_edge("C", "D")
I.add_edge("D", "2007")
I.add_edge("D", "2008")
I.add_edge("D", "E")
I.add_edge("E", "2006")
I.add_edge("E", "2009")
I.add_edge("E", "G")
I.add_edge("E", "F")
I.add_edge("G", "Stair2")
I.add_edge("Stair2", "2909")
I.add_edge("F", "H"),
I.add_edge("H", "2011")
I.add_edge("H", "2012")
I.add_edge("H", "2013")
I.add_edge("H", "2014")
I.add_edge("H", "I")
I.add_edge("I", "2016")
I.add_edge("I", "2017")
I.add_edge("I", "2018")
I.add_edge("I", "2019")
I.add_edge("I", "J")
I.add_edge("J", "2022")
I.add_edge("J", "2023")
I.add_edge("J", "2024")
I.add_edge("J", "2026")
I.add_edge("J", "K"),
I.add_edge("K", "2027")
I.add_edge("K", "2028")
I.add_edge("K", "2029")
I.add_edge("K", "2031")
I.add_edge("K", "L")
I.add_edge("L", "2907")
I.add_edge("L", "M")
I.add_edge("M", "2033")
I.add_edge("M", "2032")
I.add_edge("M", "B")

posG = nx.get_node_attributes(G, 'pos')
posI = nx.get_node_attributes(I, 'pos')
labels = nx.get_edge_attributes(G, 'weight')
labels = nx.get_edge_attributes(I, 'weight')
nx.draw(G, posG, with_labels=True)
nx.draw(I, posI, with_labels=True)
nx.draw_networkx_edge_labels(G, posG, edge_labels=labels)
nx.draw_networkx_edge_labels(I, posI, edge_labels=labels)
# print(nx.dijkstra_path(G, "Stair", "1006"))
plt.clf()


# @app.route('/<start>/<end>')
# def get_route(start, end):
#     H = nx.Graph()
#     H.clear()
#     # // Add Nodes
#     if start[0] == "1":
#         if end[0] == "1":
#             for n in G.nodes(data=True):
#                 print(n)
#                 if n[0] in nx.dijkstra_path(G, start, end):
#                     H.add_node(n[0], pos = n[1]['pos'], node_color = 'red')
#             for u, v in G.edges():
#                 if u in H.nodes and v in H.nodes:
#                     H.add_edge(u, v, weight=3, edge_color = 'black')
#             H.add_node("Cali", pos = (0,0))
#             H.add_node("Cali2", pos = (12,15))

#             image = open('/abc.png', 'rb')
#             formatGraph()
#             print(H.nodes)
#             print(image)
#             return send_from_directory(os.getcwd(), 'abc.png', as_attachment=True)

# def formatGraph():
#     img = Image.open("/abc.png")
#     img = img.convert("RGBA")

#     datas = img.getdata()

#     newData = []

#     for item in datas:
#         if item[0] == 255 and item[1] == 255 and item[2] == 255:
#             newData.append((255, 255, 255, 0))
#         else:
#             newData.append(item)

#     img.putdata(newData)
#     img.save("abc.png", "PNG")
#     print("Successful")

#     def add_margin(pil_img, top, right, bottom, left, color):
#         width, height = pil_img.size
#         new_width = width + right + left
#         new_height = height + top + bottom
#         result = Image.new(pil_img.mode, (new_width, new_height), color)
#         result.paste(pil_img, (left, top))
#         return result
#     with Image.open("abc.png") as im:
#         im_new = add_margin(im, -15, 45, -70, 120, (128, 0, 64, 1))
#         im_new.save('abc.png', quality=95)

def floorOne(H, start, end):
    for n in G.nodes(data=True):
        print(n)
        if n[0] in nx.dijkstra_path(G, start, end):
            H.add_node(n[0], pos = n[1]['pos'], node_color = 'red')
    for u, v in G.edges():
        if u in H.nodes and v in H.nodes:
            H.add_edge(u, v, weight=3, edge_color = 'black')
    H.add_node("Cali", pos = (0,0))
    H.add_node("Cali2", pos = (12,15))
    nx.draw(H, posG, with_labels=True)
    nx.draw_networkx_nodes(H, posG, nodelist = H.nodes, node_color = 'Red')
    nx.draw_networkx_nodes(H, posG, nodelist=['Cali', 'Cali2'], node_color="None")
    nx.draw_networkx_edge_labels(H, posG, edge_labels=labels)
    nx.draw(H)
def floorTwo(H, start, end):
    for n in I.nodes(data=True):
        print(n)
        if n[0] in nx.dijkstra_path(I, start, end):
            H.add_node(n[0], pos = n[1]['pos'], node_color = 'red')
    for u, v in I.edges():
        if u in H.nodes and v in H.nodes:
            H.add_edge(u, v, weight=3, edge_color = 'black')
    H.add_node("Cali", pos = (0,0))
    H.add_node("Cali2", pos = (30,15))
    nx.draw(H, posI, with_labels=True)
    nx.draw_networkx_nodes(H, posI, nodelist = H.nodes, node_color = 'Red')
    nx.draw_networkx_nodes(H, posI, nodelist=['Cali', 'Cali2'], node_color="None")
    nx.draw_networkx_edge_labels(H, posI, edge_labels=labels)
    nx.draw(H)
@app.route('/<start>/<end>')
def get_route(start, end):
    H = nx.Graph()
    H.clear()
    # // Add Nodes
    if start[0] == "1":
        if end[0] == "1":
            print("1")
            floorOne(H, start, end)
    if start[0] == "2":
        if end[0] == "2":
            print("2")
            floorTwo(H, start, end)
        # if end[0] == "2":
        #     for n in G.nodes(data=True):
        #         print(n)
        #         if n[0] in nx.dijkstra_path(G, start, "Stair"):
        #             H.add_node(n[0], pos = n[1]['pos'], node_color = 'red')
        #     for u, v in G.edges():
        #         if u in H.nodes and v in H.nodes:
        #             H.add_edge(u, v, weight=3, edge_color = 'black')

        #     for n in I.nodes(data=True):
        #         print(n)
        #         if n[0] in nx.dijkstra_path(I, "Stair", end):
        #             H.add_node(n[0], pos = n[1]['pos'], node_color = 'red')
        #     for u, v in I.edges():
        #         if u in H.nodes and v in H.nodes:
        #             H.add_edge(u, v, weight=3, edge_color = 'black')
        #     H.add_node("Cali", pos = (0,0))
        #     H.add_node("Cali2", pos = (12,15))
    print(H.nodes())
    # os.remove('/abc.png') #prevents the image from being saved twice
    plt.savefig('./static/abc.png')
    plt.clf()
    H.clear()

    img = Image.open("./static/abc.png")
    img = img.convert("RGBA")

    datas = img.getdata()

    newData = []

    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    img.save("./static/abc.png", "PNG")
    print("Successful")

    def add_margin(pil_img, top, right, bottom, left, color):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result
    with Image.open("./static/abc.png") as im:
        im_new = add_margin(im, -15, 45, -70, 120, (128, 0, 64, 1))
        im_new.save('./static/abc.png', quality=95)
    # str(nx.dijkstra_path(G, start, end)),
    image = open('./static/abc.png', 'rb')
    print(image)
    return send_from_directory('./static', 'abc.png')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
