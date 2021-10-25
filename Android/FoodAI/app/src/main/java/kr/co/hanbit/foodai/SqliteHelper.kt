package kr.co.hanbit.foodai

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.text.TextUtils

// PhotoFragment와 ListFragment를 연결하여 ListFragment의 RecyclerView에 출력될 DB
class SqliteHelper(context: Context, name: String, version: Int):SQLiteOpenHelper(context, name, null, version) {
    // Q. DB에 리스트 형태의 데이터를 저장할 수 있나??
    /* params: no: Long, datetime: String, imageUri: String, foodList: Array<String>, diary: String*/

    // 데이터베이스 최초 생성 시 호출
    override fun onCreate(db: SQLiteDatabase?) {
        // 테이블 생성 쿼리
        val create = "create table listitem" +
                "(" +
                "no integer primary key" // 순서
                "datetime text, " +      // 날짜/시간
                "imageuri text, " +      // 이미지 주소
                "foodlist text" +        // 음식 리스트
                "diary text" +           // 일기 및 기록장
                ")"
        // DB의 execSQL 메서드에 전달하여 쿼리 실행
        db?.execSQL(create)
    }
    // SqliteHelper에 전달되는 버전 정보가 변경되었을 때 현재 생성되어 있는 데이터베이스의 버전보다 높으면 호출
    override fun onUpgrade(db: SQLiteDatabase?, oldVersion: Int, newVersion: Int) {
        // TODO("Not yet implemented")
    }

    // 4개의 기본 메서드 구현
    // INSERT
    fun insertItem(item: ListItem){
        // 삽입할 데이터 작성
        val query = "insert into listitem(datetime, imageuri, foodlist, diary)" +
                "values('${item.datetime}', '${item.imageUri}', '${convertArrayToString(item.foodList, ",")}', '${item.diary}')"

        val db = writableDatabase
        db.execSQL(query)
        db.close()
    }
    // SELECT
    fun selectItem(): MutableList<ListItem>{
        val list = mutableListOf<ListItem>()

        val select = "select * from listitem" // 전체 선택
        val rd = readableDatabase
        val cursor = rd.rawQuery(select, null)
        while (cursor.moveToNext()){
            // 반복문을 돌면서 테이블에 정의된 5개의 컬럼에서 값을 꺼낸 후 변수에 저장
            val no: Long = cursor.getLong(cursor.getColumnIndex("no"))
            val datetime: String = cursor.getString(cursor.getColumnIndex("datetime"))
            val imageUri: String = cursor.getString(cursor.getColumnIndex("imageuri"))
            val foodList: Array<String> = convertStringToArray(cursor.getString(cursor.getColumnIndex("foodlist")),",")
            val diary: String = cursor.getString(cursor.getColumnIndex("diary"))

            list.add(ListItem(no, datetime, imageUri, foodList, diary))
        }
        cursor.close()
        rd.close()

        return list
    }
    // UPDATE
    fun updateItem(item: ListItem){
        // 수정할 데이터 작성
        val query = "update listitem set datetime='${item.datetime}', " +
                "imageUri='${item.imageUri}', " +
                "foodList='${convertArrayToString(item.foodList,",")}', " +
                "diary='${item.diary}'" +
                "where no = ${item.no}"
        val db = writableDatabase
        db.execSQL(query)
        db.close()
    }
    // DELETE
    fun deleteItem(item: ListItem){
        val delete = "delete from listitem where no = '${item.no}'"
        // 데이터 삭제
        val db = writableDatabase
        db.execSQL(delete)
        db.close()
        // 쓰기 전용 데이터베이스에서 삭제
        // 파라미터: 테이블명, 삭제할 조건, 조건파라미터
        val wd = writableDatabase
        wd.delete("listitem", "no = ${item.no}", null)
        wd.close()
    }

    // 리스트 - 문자열 변환 메서드
    private fun convertArrayToString(array: Array<String>, delimeter: String): String? {
        return TextUtils.join(delimeter, array)
    }
    private fun convertStringToArray(str: String, delimeter: String): Array<String> {
        return str.split(delimeter) as Array<String>
    }

}