import {Component, OnInit} from '@angular/core';
import {FormControl, FormGroup} from "@angular/forms";
import {HttpClient, HttpHeaders} from '@angular/common/http';
@Component({
  selector: 'app-tab1',
  templateUrl: 'tab1.page.html',
  styleUrls: ['tab1.page.scss']
})
export class Tab1Page implements OnInit  {
  dataForm!: FormGroup;
  constructor(private http: HttpClient) {}
  ngOnInit() {
    this.dataForm = new FormGroup({
      a0: new FormControl(1),
      a1: new FormControl(1),
      a2: new FormControl(1),
      a3: new FormControl(1),
      a4: new FormControl(1),
      a5: new FormControl(1),
      a6: new FormControl(1),
      a7: new FormControl(1),
      a8: new FormControl(1),
      a9: new FormControl(1),
      a10: new FormControl(1),
      a11: new FormControl(1),
      a12: new FormControl(1),
    });
  }
  getScore() {
    console.log("getScore")
    // let apiUrl = "http://127.0.0.1:5000/predict"
    let apiUrl = " http://127.0.0.1:5000/predict"

    let config ={ headers: new HttpHeaders().set('Access-Control-Allow-Origin', '*')
        // .set('Content-Type', 'application/json')
        // .append('Access-Control-Allow-Origin', '*')
        // .append('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE')
    }

      this.http.post(apiUrl,
      {
      "a0": this.dataForm.value.a0,
      "a1": this.dataForm.value.a1,
      "a2": this.dataForm.value.a2,
      "a3": this.dataForm.value.a3,
      "a4": this.dataForm.value.a4,
      "a5": this.dataForm.value.a5,
      "a6": this.dataForm.value.a6,
      "a7": this.dataForm.value.a7,
      "a8": this.dataForm.value.a8,
      "a9": this.dataForm.value.a9,
      "a10": this.dataForm.value.a10,
      "a11": this.dataForm.value.a11,
      "a12": this.dataForm.value.a12,
      },config).subscribe(data=>{
        console.log("response", data)
      });
  }
}
