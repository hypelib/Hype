#r "../../packages/DiffSharp.0.7.0/lib/net46/DiffSharp.dll"
#r "../../packages/Google.DataTable.Net.Wrapper.3.1.2.0/lib/Google.DataTable.Net.Wrapper.dll"
#r "../../packages/Newtonsoft.Json.7.0.1/lib/net45/Newtonsoft.Json.dll"
#r "../../packages/XPlot.GoogleCharts.1.2.1/lib/net45/XPlot.GoogleCharts.dll"
#r "../../packages/XPlot.GoogleCharts.WPF.1.2.1/lib/net45/XPlot.GoogleCharts.WPF.dll"
#r "../../src/Hype/bin/Debug/Hype.dll"

open Hype



open XPlot.GoogleCharts

let sales = ["2013", 1000; "2014", 1170; "2015", 660; "2016", 1030]
let expenses = ["2013", 400; "2014", 460; "2015", 1120; "2016", 540]

(**
Google Area Chart
=================
*)
(*** define-output:area ***)
let options =
    Options(
        title = "Company Performance",
        hAxis =
            Axis(
                title = "Year",
                titleTextStyle = TextStyle(color = "#333")
            ),
        vAxis = Axis(minValue = 0)
    )

let ch = 
    [sales; expenses]
    |> Chart.Area
    |> Chart.WithLabels ["Sales"; "Expenses"]
    |> Chart.WithOptions options
    |> Chart.WithSize (700, 500)
ch |> Chart.Show
ch.InlineHtml |> System.Windows.Forms.Clipboard.SetText

(**
<script type="text/javascript">
google.setOnLoadCallback(drawChart);
function drawChart() {
var data = new google.visualization.DataTable({"cols": [{"type": "string" ,"id": "Column 1" ,"label": "Column 1" }, {"type": "number" ,"id": "Sales" ,"label": "Sales" }, {"type": "number" ,"id": "Expenses" ,"label": "Expenses" }], "rows" : [{"c" : [{"v": "2013"}, {"v": 1000}, {"v": 400}]}, {"c" : [{"v": "2014"}, {"v": 1170}, {"v": 460}]}, {"c" : [{"v": "2015"}, {"v": 660}, {"v": 1120}]}, {"c" : [{"v": "2016"}, {"v": 1030}, {"v": 540}]}]});
var options = {"hAxis":{"title":"Year","titleTextStyle":{"color":"#333"}},"legend":{"position":"none"},"title":"Company Performance","vAxis":{"minValue":0}} 
var chart = new google.visualization.AreaChart(document.getElementById('e12cc53a-2a81-43ec-807b-629717c0f152'));
chart.draw(data, options);
}
</script>
<div id="e12cc53a-2a81-43ec-807b-629717c0f152" style="width: 700px; height: 500px; margin-left:21px"></div>
*)